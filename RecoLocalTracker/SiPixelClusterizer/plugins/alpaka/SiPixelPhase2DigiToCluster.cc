// C++ includes
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelClusterSoA/interface/alpaka/SiPixelClustersSoACollection.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigiErrorsSoACollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigisSoACollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelClusterThresholds.h"

#include "SiPixelRawToClusterKernel.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class SiPixelPhase2DigiToCluster : public stream::SynchronizingEDProducer<> {
  public:
    explicit SiPixelPhase2DigiToCluster(const edm::ParameterSet& iConfig);
    ~SiPixelPhase2DigiToCluster() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    using Algo = pixelDetails::SiPixelRawToClusterKernel<pixelTopology::Phase2>;

  private:
    void acquire(device::Event const& iEvent, device::EventSetup const& iSetup) override;
    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override;

    const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
    const edm::EDGetTokenT<edm::DetSetVector<PixelDigi>> pixelDigiToken_;

    device::EDPutToken<SiPixelDigisSoACollection> digiPutToken_;
    device::EDPutToken<SiPixelClustersSoACollection> clusterPutToken_;

    Algo Algo_;

    const SiPixelClusterThresholds clusterThresholds_;
    uint32_t nDigis_ = 0;

    SiPixelDigisSoACollection digis_d;
  };

  SiPixelPhase2DigiToCluster::SiPixelPhase2DigiToCluster(const edm::ParameterSet& iConfig)
      : geomToken_(esConsumes()),
        pixelDigiToken_(consumes<edm::DetSetVector<PixelDigi>>(iConfig.getParameter<edm::InputTag>("InputDigis"))),
        digiPutToken_(produces()),
        clusterPutToken_(produces()),
        clusterThresholds_{iConfig.getParameter<int32_t>("clusterThreshold_layer1"),
                           iConfig.getParameter<int32_t>("clusterThreshold_otherLayers"),
                           static_cast<float>(iConfig.getParameter<double>("ElectronPerADCGain")),
                           static_cast<int8_t>(iConfig.getParameter<int>("Phase2ReadoutMode")),
                           static_cast<uint16_t>(iConfig.getParameter<uint32_t>("Phase2DigiBaseline")),
                           static_cast<uint8_t>(iConfig.getParameter<uint32_t>("Phase2KinkADC"))} {}

  void SiPixelPhase2DigiToCluster::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<bool>("IncludeErrors", true);
    desc.add<int32_t>("clusterThreshold_layer1",
                      pixelClustering::clusterThresholdPhase2LayerOne);  //FIXME (fix the CUDA)
    desc.add<int32_t>("clusterThreshold_otherLayers", pixelClustering::clusterThresholdPhase2OtherLayers);
    desc.add<double>("ElectronPerADCGain", 1500.);
    desc.add<int32_t>("Phase2ReadoutMode", 3);
    desc.add<uint32_t>("Phase2DigiBaseline", 1000);
    desc.add<uint32_t>("Phase2KinkADC", 8);
    desc.add<edm::InputTag>("InputDigis", edm::InputTag("simSiPixelDigis:Pixel"));
    descriptions.addWithDefaultLabel(desc);
  }

  void SiPixelPhase2DigiToCluster::acquire(device::Event const& iEvent, device::EventSetup const& iSetup) {
    auto const& input = iEvent.get(pixelDigiToken_);

    const TrackerGeometry* geom_ = &iSetup.getData(geomToken_);

    uint32_t nDigis = 0;

    for (const auto& det : input) {
      nDigis += det.size();
    }

    if (nDigis == 0)
      return;

    nDigis_ = nDigis;
    SiPixelDigisHost digis_h(nDigis_, iEvent.queue());

    nDigis = 0;
    for (const auto& det : input) {
      unsigned int detid = det.detId();
      DetId detIdObject(detid);
      const GeomDetUnit* genericDet = geom_->idToDetUnit(detIdObject);
      auto const gind = genericDet->index();
      for (auto const& px : det) {
        digis_h.view()[nDigis].moduleId() = uint16_t(gind);

        digis_h.view()[nDigis].xx() = uint16_t(px.row());
        digis_h.view()[nDigis].yy() = uint16_t(px.column());
        digis_h.view()[nDigis].adc() = uint16_t(px.adc());

        digis_h.view()[nDigis].clus() = 0;

        digis_h.view()[nDigis].pdigi() = uint32_t(px.packedData());

        digis_h.view()[nDigis].rawIdArr() = uint32_t(detid);

        nDigis++;
      }
    }

    digis_d = SiPixelDigisSoACollection(nDigis, iEvent.queue());
    alpaka::memcpy(iEvent.queue(), digis_d.buffer(), digis_h.buffer());

    Algo_.makePhase2ClustersAsync(iEvent.queue(), clusterThresholds_, digis_d.view(), nDigis_);
  }

  void SiPixelPhase2DigiToCluster::produce(device::Event& iEvent, device::EventSetup const& iSetup) {
    if (nDigis_ == 0) {
      SiPixelClustersSoACollection clusters_d{pixelTopology::Phase2::numberOfModules, iEvent.queue()};
      SiPixelDigisSoACollection digis_d_zero{nDigis_, iEvent.queue()};
      iEvent.emplace(digiPutToken_, std::move(digis_d_zero));
      iEvent.emplace(clusterPutToken_, std::move(clusters_d));
      return;
    }

    digis_d.setNModules(Algo_.nModules());
    iEvent.emplace(digiPutToken_, std::move(digis_d));
    iEvent.emplace(clusterPutToken_, Algo_.getClusters());
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// define as framework plugin
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(SiPixelPhase2DigiToCluster);
