#include "CUDADataFormats/Common/interface/Product.h"
#include "CUDADataFormats/SiPixelCluster/interface/SiPixelClustersCUDA.h"
#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDA.h"
#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigiErrorsCUDA.h"
#include "CalibTracker/Records/interface/SiPixelGainCalibrationForHLTGPURcd.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationForHLTGPU.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "EventFilter/SiPixelRawToDigi/interface/PixelDataFormatter.h"
#include "EventFilter/SiPixelRawToDigi/interface/PixelUnpackingRegions.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelFedCablingMapGPUWrapper.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

#include "SiPixelRawToClusterGPUKernel.h"

#include <memory>
#include <string>
#include <vector>

class SiPixelRawToClusterCUDA : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit SiPixelRawToClusterCUDA(const edm::ParameterSet& iConfig);
  ~SiPixelRawToClusterCUDA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void acquire(const edm::Event& iEvent,
               const edm::EventSetup& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::EDGetTokenT<FEDRawDataCollection> rawGetToken_;

  edm::EDPutTokenT<cms::cuda::Product<SiPixelDigisCUDA>> digiPutToken_;
  edm::EDPutTokenT<cms::cuda::Product<SiPixelDigiErrorsCUDA>> digiErrorPutToken_;
  edm::EDPutTokenT<cms::cuda::Product<SiPixelClustersCUDA>> clusterPutToken_;

  cms::cuda::ContextState ctxState_;

  edm::ESWatcher<SiPixelFedCablingMapRcd> recordWatcher_;
  edm::ESGetToken<SiPixelFedCablingMapGPUWrapper, CkfComponentsRecord> gpuMapToken_;
  edm::ESGetToken<SiPixelGainCalibrationForHLTGPU, SiPixelGainCalibrationForHLTGPURcd> gainsToken_;
  edm::ESGetToken<SiPixelFedCablingMap, SiPixelFedCablingMapRcd> cablingMapToken_;

  std::unique_ptr<SiPixelFedCablingTree> cabling_;
  std::vector<unsigned int> fedIds_;
  const SiPixelFedCablingMap* cablingMap_ = nullptr;
  std::unique_ptr<PixelUnpackingRegions> regions_;

  pixelgpudetails::SiPixelRawToClusterGPUKernel gpuAlgo_;
  std::unique_ptr<pixelgpudetails::SiPixelRawToClusterGPUKernel::WordFedAppender> wordFedAppender_;
  PixelDataFormatter::Errors errors_;

  const bool includeErrors_;
  const bool useQuality_;
  const bool usePilotBlade_;
};

SiPixelRawToClusterCUDA::SiPixelRawToClusterCUDA(const edm::ParameterSet& iConfig)
    : rawGetToken_(consumes<FEDRawDataCollection>(iConfig.getParameter<edm::InputTag>("InputLabel"))),
      digiPutToken_(produces<cms::cuda::Product<SiPixelDigisCUDA>>()),
      clusterPutToken_(produces<cms::cuda::Product<SiPixelClustersCUDA>>()),
      gpuMapToken_(esConsumes<SiPixelFedCablingMapGPUWrapper, CkfComponentsRecord>()),
      gainsToken_(esConsumes<SiPixelGainCalibrationForHLTGPU, SiPixelGainCalibrationForHLTGPURcd>()),
      cablingMapToken_(esConsumes<SiPixelFedCablingMap, SiPixelFedCablingMapRcd>(
          edm::ESInputTag("", iConfig.getParameter<std::string>("CablingMapLabel")))),
      includeErrors_(iConfig.getParameter<bool>("IncludeErrors")),
      useQuality_(iConfig.getParameter<bool>("UseQualityInfo")),
      usePilotBlade_(iConfig.getParameter<bool>("UsePilotBlade"))  // Control the usage of pilot-blade data, FED=40
{
  if (includeErrors_) {
    digiErrorPutToken_ = produces<cms::cuda::Product<SiPixelDigiErrorsCUDA>>();
  }

  // regions
  if (!iConfig.getParameter<edm::ParameterSet>("Regions").getParameterNames().empty()) {
    regions_ = std::make_unique<PixelUnpackingRegions>(iConfig, consumesCollector());
  }

  if (usePilotBlade_)
    edm::LogInfo("SiPixelRawToCluster") << " Use pilot blade data (FED 40)";

  edm::Service<CUDAService> cs;
  if (cs->enabled()) {
    wordFedAppender_ = std::make_unique<pixelgpudetails::SiPixelRawToClusterGPUKernel::WordFedAppender>();
  }
}

void SiPixelRawToClusterCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("IncludeErrors", true);
  desc.add<bool>("UseQualityInfo", false);
  desc.add<bool>("UsePilotBlade", false)->setComment("##  Use pilot blades");
  desc.add<edm::InputTag>("InputLabel", edm::InputTag("rawDataCollector"));
  {
    edm::ParameterSetDescription psd0;
    psd0.addOptional<std::vector<edm::InputTag>>("inputs");
    psd0.addOptional<std::vector<double>>("deltaPhi");
    psd0.addOptional<std::vector<double>>("maxZ");
    psd0.addOptional<edm::InputTag>("beamSpot");
    desc.add<edm::ParameterSetDescription>("Regions", psd0)
        ->setComment("## Empty Regions PSet means complete unpacking");
  }
  desc.add<std::string>("CablingMapLabel", "")->setComment("CablingMap label");  //Tav
  descriptions.addWithDefaultLabel(desc);
}

void SiPixelRawToClusterCUDA::acquire(const edm::Event& iEvent,
                                      const edm::EventSetup& iSetup,
                                      edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  cms::cuda::ScopedContextAcquire ctx{iEvent.streamID(), std::move(waitingTaskHolder), ctxState_};

  auto hgpuMap = iSetup.getHandle(gpuMapToken_);
  if (hgpuMap->hasQuality() != useQuality_) {
    throw cms::Exception("LogicError")
        << "UseQuality of the module (" << useQuality_
        << ") differs the one from SiPixelFedCablingMapGPUWrapper. Please fix your configuration.";
  }
  // get the GPU product already here so that the async transfer can begin
  const auto* gpuMap = hgpuMap->getGPUProductAsync(ctx.stream());

  auto hgains = iSetup.getHandle(gainsToken_);
  // get the GPU product already here so that the async transfer can begin
  const auto* gpuGains = hgains->getGPUProductAsync(ctx.stream());

  cms::cuda::device::unique_ptr<unsigned char[]> modulesToUnpackRegional;
  const unsigned char* gpuModulesToUnpack;

  if (regions_) {
    regions_->run(iEvent, iSetup);
    LogDebug("SiPixelRawToCluster") << "region2unpack #feds: " << regions_->nFEDs();
    LogDebug("SiPixelRawToCluster") << "region2unpack #modules (BPIX,EPIX,total): " << regions_->nBarrelModules() << " "
                                    << regions_->nForwardModules() << " " << regions_->nModules();
    modulesToUnpackRegional = hgpuMap->getModToUnpRegionalAsync(*(regions_->modulesToUnpack()), ctx.stream());
    gpuModulesToUnpack = modulesToUnpackRegional.get();
  } else {
    gpuModulesToUnpack = hgpuMap->getModToUnpAllAsync(ctx.stream());
  }

  // initialize cabling map or update if necessary
  if (recordWatcher_.check(iSetup)) {
    // cabling map, which maps online address (fed->link->ROC->local pixel) to offline (DetId->global pixel)
    auto cablingMap = iSetup.getTransientHandle(cablingMapToken_);
    cablingMap_ = cablingMap.product();
    fedIds_ = cablingMap->fedIds();
    cabling_ = cablingMap->cablingTree();
    LogDebug("map version:") << cabling_->version();
  }

  const auto& buffers = iEvent.get(rawGetToken_);

  errors_.clear();

  // GPU specific: Data extraction for RawToDigi GPU
  unsigned int wordCounterGPU = 0;
  unsigned int fedCounter = 0;
  bool errorsInEvent = false;

  // In CPU algorithm this loop is part of PixelDataFormatter::interpretRawData()
  ErrorChecker errorcheck;
  for (int fedId : fedIds_) {
    if (!usePilotBlade_ && (fedId == 40))
      continue;  // skip pilot blade data
    if (regions_ && !regions_->mayUnpackFED(fedId))
      continue;

    // for GPU
    // first 150 index stores the fedId and next 150 will store the
    // start index of word in that fed
    assert(fedId >= 1200);
    fedCounter++;

    // get event data for this fed
    const FEDRawData& rawData = buffers.FEDData(fedId);

    // GPU specific
    int nWords = rawData.size() / sizeof(cms_uint64_t);
    if (nWords == 0) {
      continue;
    }

    // check CRC bit
    const cms_uint64_t* trailer = reinterpret_cast<const cms_uint64_t*>(rawData.data()) + (nWords - 1);
    if (not errorcheck.checkCRC(errorsInEvent, fedId, trailer, errors_)) {
      continue;
    }

    // check headers
    const cms_uint64_t* header = reinterpret_cast<const cms_uint64_t*>(rawData.data());
    header--;
    bool moreHeaders = true;
    while (moreHeaders) {
      header++;
      bool headerStatus = errorcheck.checkHeader(errorsInEvent, fedId, header, errors_);
      moreHeaders = headerStatus;
    }

    // check trailers
    bool moreTrailers = true;
    trailer++;
    while (moreTrailers) {
      trailer--;
      bool trailerStatus = errorcheck.checkTrailer(errorsInEvent, fedId, nWords, trailer, errors_);
      moreTrailers = trailerStatus;
    }

    const cms_uint32_t* bw = (const cms_uint32_t*)(header + 1);
    const cms_uint32_t* ew = (const cms_uint32_t*)(trailer);

    assert(0 == (ew - bw) % 2);
    wordFedAppender_->initializeWordFed(fedId, wordCounterGPU, bw, (ew - bw));
    wordCounterGPU += (ew - bw);

  }  // end of for loop

  gpuAlgo_.makeClustersAsync(gpuMap,
                             gpuModulesToUnpack,
                             gpuGains,
                             *wordFedAppender_,
                             std::move(errors_),
                             wordCounterGPU,
                             fedCounter,
                             useQuality_,
                             includeErrors_,
                             edm::MessageDrop::instance()->debugEnabled,
                             ctx.stream());
}

void SiPixelRawToClusterCUDA::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  cms::cuda::ScopedContextProduce ctx{ctxState_};

  auto tmp = gpuAlgo_.getResults();
  ctx.emplace(iEvent, digiPutToken_, std::move(tmp.first));
  ctx.emplace(iEvent, clusterPutToken_, std::move(tmp.second));
  if (includeErrors_) {
    ctx.emplace(iEvent, digiErrorPutToken_, gpuAlgo_.getErrors());
  }
}

// define as framework plugin
DEFINE_FWK_MODULE(SiPixelRawToClusterCUDA);
