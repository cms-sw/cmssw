#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "CalibTracker/Records/interface/SiPixelGainCalibrationForHLTSoARcd.h"
#include "CalibTracker/Records/interface/SiPixelMappingSoARecord.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include "CondFormats/SiPixelObjects/interface/alpaka/SiPixelGainCalibrationForHLTDevice.h"
#include "CondFormats/SiPixelObjects/interface/alpaka/SiPixelMappingDevice.h"
#include "CondFormats/SiPixelObjects/interface/alpaka/SiPixelMappingUtilities.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/SiPixelClusterSoA/interface/alpaka/SiPixelClustersSoACollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigiErrorsSoACollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigisSoACollection.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelFormatterErrors.h"
#include "EventFilter/SiPixelRawToDigi/interface/PixelDataFormatter.h"
#include "EventFilter/SiPixelRawToDigi/interface/PixelUnpackingRegions.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelClusterThresholds.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelImageSoA.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelImageDevice.h"

#include "SiPixelRawToClusterKernel.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename TrackerTraits>
  class SiPixelRawToCluster : public stream::SynchronizingEDProducer<> {
  public:
    explicit SiPixelRawToCluster(const edm::ParameterSet& iConfig);
    ~SiPixelRawToCluster() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    using Algo = pixelDetails::SiPixelRawToClusterKernel<TrackerTraits>;

  private:
    void acquire(device::Event const& iEvent, device::EventSetup const& iSetup) override;
    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override;

    edm::EDGetTokenT<FEDRawDataCollection> rawGetToken_;
    edm::EDPutTokenT<SiPixelFormatterErrors> fmtErrorToken_;
    device::EDPutToken<SiPixelDigisSoACollection> digiPutToken_;
    device::EDPutToken<SiPixelDigiErrorsSoACollection> digiErrorPutToken_;
    device::EDPutToken<SiPixelClustersSoACollection> clusterPutToken_;

    edm::ESWatcher<SiPixelFedCablingMapRcd> recordWatcher_;
    const device::ESGetToken<SiPixelMappingDevice, SiPixelMappingSoARecord> mapToken_;
    const device::ESGetToken<SiPixelGainCalibrationForHLTDevice, SiPixelGainCalibrationForHLTSoARcd> gainsToken_;
    const edm::ESGetToken<SiPixelFedCablingMap, SiPixelFedCablingMapRcd> cablingMapToken_;

    std::unique_ptr<SiPixelFedCablingTree> cabling_;
    std::vector<unsigned int> fedIds_;
    const SiPixelFedCablingMap* cablingMap_ = nullptr;
    std::unique_ptr<PixelUnpackingRegions> regions_;

    Algo Algo_;
    PixelDataFormatter::Errors errors_;

    const bool includeErrors_;
    const bool useQuality_;
    uint32_t nDigis_;
    const SiPixelClusterThresholds clusterThresholds_;
    std::optional<SiPixelImageDevice> images_;
  };

  template <typename TrackerTraits>
  SiPixelRawToCluster<TrackerTraits>::SiPixelRawToCluster(const edm::ParameterSet& iConfig)
      : SynchronizingEDProducer(iConfig),
        rawGetToken_(consumes(iConfig.getParameter<edm::InputTag>("InputLabel"))),
        digiPutToken_(produces()),
        clusterPutToken_(produces()),
        mapToken_(esConsumes()),
        gainsToken_(esConsumes()),
        cablingMapToken_(esConsumes<SiPixelFedCablingMap, SiPixelFedCablingMapRcd>(
            edm::ESInputTag("", iConfig.getParameter<std::string>("CablingMapLabel")))),
        includeErrors_(iConfig.getParameter<bool>("IncludeErrors")),
        useQuality_(iConfig.getParameter<bool>("UseQualityInfo")),
        clusterThresholds_{iConfig.getParameter<int32_t>("clusterThreshold_layer1"),
                           iConfig.getParameter<int32_t>("clusterThreshold_otherLayers"),
                           static_cast<float>(iConfig.getParameter<double>("VCaltoElectronGain")),
                           static_cast<float>(iConfig.getParameter<double>("VCaltoElectronGain_L1")),
                           static_cast<float>(iConfig.getParameter<double>("VCaltoElectronOffset")),
                           static_cast<float>(iConfig.getParameter<double>("VCaltoElectronOffset_L1"))} {
    if (includeErrors_) {
      digiErrorPutToken_ = produces();
      fmtErrorToken_ = produces();
    }

    // regions
    if (!iConfig.getParameter<edm::ParameterSet>("Regions").getParameterNames().empty()) {
      regions_ = std::make_unique<PixelUnpackingRegions>(iConfig, consumesCollector());
    }
  }

  template <typename TrackerTraits>
  void SiPixelRawToCluster<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<bool>("IncludeErrors", true);
    desc.add<bool>("UseQualityInfo", false);
    // Note: this parameter is obsolete: it is ignored and will have no effect.
    // It is kept to avoid breaking older configurations, and will not be printed in the generated cfi.py file.
    desc.addOptionalNode(edm::ParameterDescription<uint32_t>("MaxFEDWords", 0, true), false)
        ->setComment("This parameter is obsolete and will be ignored.");
    desc.add<int32_t>("clusterThreshold_layer1", pixelClustering::clusterThresholdLayerOne);
    desc.add<int32_t>("clusterThreshold_otherLayers", pixelClustering::clusterThresholdOtherLayers);
    desc.add<double>("VCaltoElectronGain", 47.f);
    desc.add<double>("VCaltoElectronGain_L1", 50.f);
    desc.add<double>("VCaltoElectronOffset", -60.f);
    desc.add<double>("VCaltoElectronOffset_L1", -670.f);

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

  template <typename TrackerTraits>
  void SiPixelRawToCluster<TrackerTraits>::acquire(device::Event const& iEvent, device::EventSetup const& iSetup) {
    auto const& hMap = iSetup.getData(mapToken_);
    auto const& dGains = iSetup.getData(gainsToken_);

    // initialize cabling map or update if necessary
    if (recordWatcher_.check(iSetup)) {
      // cabling map, which maps online address (fed->link->ROC->local pixel) to offline (DetId->global pixel)
      cablingMap_ = &iSetup.getData(cablingMapToken_);
      fedIds_ = cablingMap_->fedIds();
      cabling_ = cablingMap_->cablingTree();
      LogDebug("map version:") << cablingMap_->version();
    }

    // if used, the buffer is guaranteed to stay alive until the after the execution of makePhase1ClustersAsync completes
    std::optional<cms::alpakatools::device_buffer<Device, unsigned char[]>> modulesToUnpackRegional;
    const unsigned char* modulesToUnpack;
    if (regions_) {
      regions_->run(iEvent, iSetup);
      LogDebug("SiPixelRawToCluster") << "region2unpack #feds: " << regions_->nFEDs();
      LogDebug("SiPixelRawToCluster") << "region2unpack #modules (BPIX,EPIX,total): " << regions_->nBarrelModules()
                                      << " " << regions_->nForwardModules() << " " << regions_->nModules();

      modulesToUnpackRegional = SiPixelMappingUtilities::getModToUnpRegionalAsync(
          *(regions_->modulesToUnpack()), cabling_.get(), fedIds_, iEvent.queue());
      modulesToUnpack = modulesToUnpackRegional->data();
    } else {
      modulesToUnpack = hMap->modToUnpDefault();
    }

    const auto& buffers = iEvent.get(rawGetToken_);

    errors_.clear();

    // GPU specific: Data extraction for RawToDigi GPU
    unsigned int wordCounter = 0;
    unsigned int fedCounter = 0;
    bool errorsInEvent = false;
    std::vector<unsigned int> index(fedIds_.size(), 0);
    std::vector<cms_uint32_t const*> start(fedIds_.size(), nullptr);
    std::vector<ptrdiff_t> words(fedIds_.size(), 0);
    // In CPU algorithm this loop is part of PixelDataFormatter::interpretRawData()
    ErrorChecker errorcheck;
    for (uint32_t i = 0; i < fedIds_.size(); ++i) {
      const int fedId = fedIds_[i];
      if (regions_ && !regions_->mayUnpackFED(fedId))
        continue;

      // for GPU
      // first 150 index stores the fedId and next 150 will store the
      // start index of word in that fed
      assert(fedId >= FEDNumbering::MINSiPixeluTCAFEDID);
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
      index[i] = wordCounter;
      start[i] = bw;
      words[i] = (ew - bw);
      wordCounter += (ew - bw);

    }  // end of for loop
    nDigis_ = wordCounter;
    if (nDigis_ == 0)
      return;

    // copy the FED data to a single cpu buffer
    pixelDetails::WordFedAppender wordFedAppender(iEvent.queue(), nDigis_);
    for (uint32_t i = 0; i < fedIds_.size(); ++i) {
      wordFedAppender.initializeWordFed(fedIds_[i], index[i], start[i], words[i]);
    }
    images_ = SiPixelImageDevice(pixelTopology::Phase1::numberOfModules,iEvent.queue());

    Algo_.makePhase1ClustersAsync(iEvent.queue(),
                                  clusterThresholds_,
				  images_->view(),
                                  hMap.const_view(),
                                  modulesToUnpack,
                                  dGains.const_view(),
                                  wordFedAppender,
                                  wordCounter,
                                  fedCounter,
                                  useQuality_,
                                  includeErrors_,
                                  edm::MessageDrop::instance()->debugEnabled);
  }

  template <typename TrackerTraits>
  void SiPixelRawToCluster<TrackerTraits>::produce(device::Event& iEvent, device::EventSetup const& iSetup) {
    if (nDigis_ == 0) {
      // Cannot use the default constructor here, as it would not allocate memory.
      // In the case of no digis, clusters_d are not being instantiated, but are
      // still used downstream to initialize TrackingRecHitSoADevice. If there
      // are no valid pointers to clusters' Collection columns, instantiation
      // of TrackingRecHits fail. Example: workflow 11604.0

      iEvent.emplace(digiPutToken_, 0, iEvent.queue());
      iEvent.emplace(clusterPutToken_, pixelTopology::Phase1::numberOfModules, iEvent.queue());
      if (includeErrors_) {
        iEvent.emplace(digiErrorPutToken_, 0, iEvent.queue());
        iEvent.emplace(fmtErrorToken_);
      }
      return;
    }

    iEvent.emplace(digiPutToken_, Algo_.getDigis());
    iEvent.emplace(clusterPutToken_, Algo_.getClusters());
    if (includeErrors_) {
      iEvent.emplace(digiErrorPutToken_, Algo_.getErrors());
      iEvent.emplace(fmtErrorToken_, std::move(errors_));
    }
  }

  using SiPixelRawToClusterPhase1 = SiPixelRawToCluster<pixelTopology::Phase1>;
  using SiPixelRawToClusterHIonPhase1 = SiPixelRawToCluster<pixelTopology::HIonPhase1>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// define as framework plugin
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(SiPixelRawToClusterPhase1);
DEFINE_FWK_ALPAKA_MODULE(SiPixelRawToClusterHIonPhase1);
