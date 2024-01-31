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
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "EventFilter/SiPixelRawToDigi/interface/PixelDataFormatter.h"
#include "EventFilter/SiPixelRawToDigi/interface/PixelUnpackingRegions.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelClusterThresholds.h"

// local includes
#include "SiPixelRawToClusterGPUKernel.h"

class SiPixelPhase2DigiToClusterCUDA : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit SiPixelPhase2DigiToClusterCUDA(const edm::ParameterSet& iConfig);
  ~SiPixelPhase2DigiToClusterCUDA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  using GPUAlgo = pixelgpudetails::SiPixelRawToClusterGPUKernel<pixelTopology::Phase2>;

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

  GPUAlgo gpuAlgo_;

  const bool includeErrors_;
  const SiPixelClusterThresholds clusterThresholds_;
  uint32_t nDigis_;
};

SiPixelPhase2DigiToClusterCUDA::SiPixelPhase2DigiToClusterCUDA(const edm::ParameterSet& iConfig)
    : geomToken_(esConsumes()),
      pixelDigiToken_(consumes<edm::DetSetVector<PixelDigi>>(iConfig.getParameter<edm::InputTag>("InputDigis"))),
      digiPutToken_(produces<cms::cuda::Product<SiPixelDigisCUDA>>()),
      clusterPutToken_(produces<cms::cuda::Product<SiPixelClustersCUDA>>()),
      includeErrors_(iConfig.getParameter<bool>("IncludeErrors")),
      clusterThresholds_{iConfig.getParameter<int32_t>("clusterThreshold_layer1"),
                         iConfig.getParameter<int32_t>("clusterThreshold_otherLayers"),
                         static_cast<float>(iConfig.getParameter<double>("ElectronPerADCGain")),
                         static_cast<int8_t>(iConfig.getParameter<int>("Phase2ReadoutMode")),
                         static_cast<uint16_t>(iConfig.getParameter<uint32_t>("Phase2DigiBaseline")),
                         static_cast<uint8_t>(iConfig.getParameter<uint32_t>("Phase2KinkADC"))} {
  if (includeErrors_) {
    digiErrorPutToken_ = produces<cms::cuda::Product<SiPixelDigiErrorsCUDA>>();
  }
}

void SiPixelPhase2DigiToClusterCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<bool>("IncludeErrors", true);
  desc.add<int32_t>("clusterThreshold_layer1", 4000);
  desc.add<int32_t>("clusterThreshold_otherLayers", 4000);
  desc.add<double>("ElectronPerADCGain", 1500);
  desc.add<int32_t>("Phase2ReadoutMode", 3);
  desc.add<uint32_t>("Phase2DigiBaseline", 1000);
  desc.add<uint32_t>("Phase2KinkADC", 8);
  desc.add<edm::InputTag>("InputDigis", edm::InputTag("simSiPixelDigis:Pixel"));
  descriptions.addWithDefaultLabel(desc);
}

void SiPixelPhase2DigiToClusterCUDA::acquire(const edm::Event& iEvent,
                                             const edm::EventSetup& iSetup,
                                             edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  cms::cuda::ScopedContextAcquire ctx{iEvent.streamID(), std::move(waitingTaskHolder), ctxState_};

  auto const& input = iEvent.get(pixelDigiToken_);

  const TrackerGeometry* geom_ = &iSetup.getData(geomToken_);

  nDigis_ = 0;

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
      moduleIds[nDigis_] = uint16_t(gind);

      xDigis[nDigis_] = uint16_t(px.row());
      yDigis[nDigis_] = uint16_t(px.column());
      adcDigis[nDigis_] = uint16_t(px.adc());

      packedData[nDigis_] = uint32_t(px.packedData());

      rawIds[nDigis_] = uint32_t(detid);

      nDigis_++;
    }
  }

  if (nDigis_ == 0)
    return;

  gpuAlgo_.makePhase2ClustersAsync(clusterThresholds_,
                                   moduleIds.get(),
                                   xDigis.get(),
                                   yDigis.get(),
                                   adcDigis.get(),
                                   packedData.get(),
                                   rawIds.get(),
                                   nDigis_,
                                   ctx.stream());
}

void SiPixelPhase2DigiToClusterCUDA::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  cms::cuda::ScopedContextProduce ctx{ctxState_};

  if (nDigis_ == 0) {
    ctx.emplace(iEvent, digiPutToken_, nDigis_, ctx.stream());
    ctx.emplace(iEvent, clusterPutToken_, pixelTopology::Phase2::numberOfModules, ctx.stream());
    if (includeErrors_) {
      ctx.emplace(iEvent, digiErrorPutToken_, SiPixelDigiErrorsCUDA{});
    }
    return;
  }

  auto tmp = gpuAlgo_.getResults();
  ctx.emplace(iEvent, digiPutToken_, std::move(tmp.first));
  ctx.emplace(iEvent, clusterPutToken_, std::move(tmp.second));
  if (includeErrors_) {
    ctx.emplace(iEvent, digiErrorPutToken_, gpuAlgo_.getErrors());
  }
}

// define as framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiPixelPhase2DigiToClusterCUDA);
