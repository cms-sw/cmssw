// C++ includes
#include <memory>
#include <string>
#include <vector>

// CMSSW includes
#include "CUDADataFormats/Common/interface/Product.h"
#include "CUDADataFormats/SiPixelCluster/interface/SiPixelClustersCUDA.h"
#include "CUDADataFormats/SiPixelCluster/interface/gpuClusteringConstants.h"
#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigiErrorsCUDA.h"
#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDA.h"
#include "CalibTracker/Records/interface/SiPixelGainCalibrationForHLTGPURcd.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelROCsStatusAndMappingWrapper.h"
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
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAInterface.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

// local includes
#include "SiPixelClusterThresholds.h"
#include "SiPixelRawToClusterGPUKernel.h"

class SiPixelPhase2DigiToClusterCUDA : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit SiPixelPhase2DigiToClusterCUDA(const edm::ParameterSet& iConfig);
  ~SiPixelPhase2DigiToClusterCUDA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void acquire(const edm::Event& iEvent,
               const edm::EventSetup& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::EDGetTokenT<edm::DetSetVector<PixelDigi>> pixelDigiToken_;

  edm::EDPutTokenT<cms::cuda::Product<SiPixelDigisCUDA>> digiPutToken_;
  edm::EDPutTokenT<cms::cuda::Product<SiPixelDigiErrorsCUDA>> digiErrorPutToken_;
  edm::EDPutTokenT<cms::cuda::Product<SiPixelClustersCUDA>> clusterPutToken_;

  cms::cuda::ContextState ctxState_;

  pixelgpudetails::SiPixelRawToClusterGPUKernel gpuAlgo_;
  std::unique_ptr<pixelgpudetails::SiPixelRawToClusterGPUKernel::WordFedAppender> wordFedAppender_;

  const bool includeErrors_;
  const SiPixelClusterThresholds clusterThresholds_;
};

SiPixelPhase2DigiToClusterCUDA::SiPixelPhase2DigiToClusterCUDA(const edm::ParameterSet& iConfig)
    : geomToken_(esConsumes()),
      pixelDigiToken_(consumes<edm::DetSetVector<PixelDigi>>(iConfig.getParameter<edm::InputTag>("InputDigis"))),
      digiPutToken_(produces<cms::cuda::Product<SiPixelDigisCUDA>>()),
      clusterPutToken_(produces<cms::cuda::Product<SiPixelClustersCUDA>>()),
      includeErrors_(iConfig.getParameter<bool>("IncludeErrors")),
      clusterThresholds_{iConfig.getParameter<int32_t>("clusterThreshold_layer1"),
                         iConfig.getParameter<int32_t>("clusterThreshold_otherLayers")} {
  if (includeErrors_) {
    digiErrorPutToken_ = produces<cms::cuda::Product<SiPixelDigiErrorsCUDA>>();
  }
}

void SiPixelPhase2DigiToClusterCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<bool>("IncludeErrors", true);
  desc.add<int32_t>("clusterThreshold_layer1", kSiPixelClusterThresholdsDefaultPhase2.layer1);
  desc.add<int32_t>("clusterThreshold_otherLayers", kSiPixelClusterThresholdsDefaultPhase2.otherLayers);
  desc.add<edm::InputTag>("InputDigis", edm::InputTag("simSiPixelDigis:Pixel"));
  descriptions.addWithDefaultLabel(desc);
}

void SiPixelPhase2DigiToClusterCUDA::acquire(const edm::Event& iEvent,
                                             const edm::EventSetup& iSetup,
                                             edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  cms::cuda::ScopedContextAcquire ctx{iEvent.streamID(), std::move(waitingTaskHolder), ctxState_};

  auto const& input = iEvent.get(pixelDigiToken_);

  const TrackerGeometry* geom_ = &iSetup.getData(geomToken_);

  uint32_t nDigis = 0;

  auto xDigis = cms::cuda::make_host_unique<uint16_t[]>(gpuClustering::maxNumDigis, ctx.stream());
  auto yDigis = cms::cuda::make_host_unique<uint16_t[]>(gpuClustering::maxNumDigis, ctx.stream());
  auto adcDigis = cms::cuda::make_host_unique<uint16_t[]>(gpuClustering::maxNumDigis, ctx.stream());
  auto moduleIds = cms::cuda::make_host_unique<uint16_t[]>(gpuClustering::maxNumDigis, ctx.stream());
  auto packedData = cms::cuda::make_host_unique<uint32_t[]>(gpuClustering::maxNumDigis, ctx.stream());
  auto rawIds = cms::cuda::make_host_unique<uint32_t[]>(gpuClustering::maxNumDigis, ctx.stream());

  for (auto DSViter = input.begin(); DSViter != input.end(); DSViter++) {
    unsigned int detid = DSViter->detId();
    DetId detIdObject(detid);
    const GeomDetUnit* genericDet = geom_->idToDetUnit(detIdObject);
    auto const gind = genericDet->index();
    for (auto const& px : *DSViter) {
      moduleIds[nDigis] = uint16_t(gind);

      xDigis[nDigis] = uint16_t(px.row());
      yDigis[nDigis] = uint16_t(px.column());
      adcDigis[nDigis] = uint16_t(px.adc());

      packedData[nDigis] = uint32_t(px.packedData());

      rawIds[nDigis] = uint32_t(detid);

      nDigis++;
    }
  }

  gpuAlgo_.makePhase2ClustersAsync(clusterThresholds_,
                                   moduleIds.get(),
                                   xDigis.get(),
                                   yDigis.get(),
                                   adcDigis.get(),
                                   packedData.get(),
                                   rawIds.get(),
                                   nDigis,
                                   ctx.stream());
}

void SiPixelPhase2DigiToClusterCUDA::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  cms::cuda::ScopedContextProduce ctx{ctxState_};

  auto tmp = gpuAlgo_.getResults();
  ctx.emplace(iEvent, digiPutToken_, std::move(tmp.first));
  ctx.emplace(iEvent, clusterPutToken_, std::move(tmp.second));
  if (includeErrors_) {
    ctx.emplace(iEvent, digiErrorPutToken_, gpuAlgo_.getErrors());
  }
}

// define as framework plugin
DEFINE_FWK_MODULE(SiPixelPhase2DigiToClusterCUDA);
