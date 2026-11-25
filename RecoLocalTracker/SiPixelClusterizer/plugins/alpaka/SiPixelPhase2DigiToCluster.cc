// C++ includes
#include <algorithm>
#include <memory>
#include <optional>
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
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CommonTopologies/interface/GeomDetEnumerators.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelClusterThresholds.h"

#include "SiPixelMorphingConfig.h"
#include "SiPixelMorphingRegions.h"
#include "SiPixelRawToClusterKernel.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class SiPixelPhase2DigiToCluster : public stream::SynchronizingEDProducer<edm::stream::WatchRuns> {
  public:
    explicit SiPixelPhase2DigiToCluster(const edm::ParameterSet& iConfig);
    ~SiPixelPhase2DigiToCluster() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    using Algo = pixelDetails::SiPixelRawToClusterKernel<pixelTopology::Phase2>;

  private:
    void acquire(device::Event const& iEvent, device::EventSetup const& iSetup) override;
    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override;
    void beginRun(edm::Run const&, edm::EventSetup const& iSetup) override;

    const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
    const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomTokenBeginRun_;  // For BeginRun
    const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;

    const edm::EDGetTokenT<edm::DetSetVector<PixelDigi>> pixelDigiToken_;
    const device::EDPutToken<SiPixelDigisSoACollection> digiPutToken_;
    const device::EDPutToken<SiPixelClustersSoACollection> clusterPutToken_;
    const SiPixelClusterThresholds clusterThresholds_;

    SiPixelMorphingConfig digiMorphingConfig_;
    const std::vector<siPixelMorphing::region> theBarrelRegions_;
    const std::vector<siPixelMorphing::region> theEndcapRegions_;
    // sorted raw ids of the modules where the digi morphing is applied, and their sizes (rows << 16 | columns)
    std::vector<uint32_t> morphingModulesHost_;
    std::vector<uint32_t> morphingModuleSizesHost_;
    bool morphingModulesChanged_ = false;
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> morphingModulesDevice_;
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> morphingModuleSizesDevice_;

    Algo algo_;
    uint32_t nDigis_ = 0;
    std::optional<SiPixelDigisSoACollection> digis_d_;
    mutable uint32_t offsetBPIX2_ = pixelTopology::Phase2::layerStart[1];
  };

  SiPixelPhase2DigiToCluster::SiPixelPhase2DigiToCluster(const edm::ParameterSet& iConfig)
      : SynchronizingEDProducer(iConfig),
        geomToken_(esConsumes()),
        geomTokenBeginRun_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>()),
        topoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::BeginRun>()),
        pixelDigiToken_(consumes<edm::DetSetVector<PixelDigi>>(iConfig.getParameter<edm::InputTag>("InputDigis"))),
        digiPutToken_(produces()),
        clusterPutToken_(produces()),
        clusterThresholds_{iConfig.getParameter<int32_t>("clusterThreshold_layer1"),
                           iConfig.getParameter<int32_t>("clusterThreshold_otherLayers"),
                           static_cast<float>(iConfig.getParameter<double>("ElectronPerADCGain")),
                           static_cast<int8_t>(iConfig.getParameter<int>("Phase2ReadoutMode")),
                           static_cast<uint16_t>(iConfig.getParameter<uint32_t>("Phase2DigiBaseline")),
                           static_cast<uint8_t>(iConfig.getParameter<uint32_t>("Phase2KinkADC"))},
        theBarrelRegions_(
            siPixelMorphing::parseRegions(iConfig.getParameter<std::vector<std::string>>("barrelRegions"), 3)),
        theEndcapRegions_(
            siPixelMorphing::parseRegions(iConfig.getParameter<std::vector<std::string>>("endcapRegions"), 4)) {
    digiMorphingConfig_.applyDigiMorphing = iConfig.getParameter<bool>("DoDigiMorphing");
    digiMorphingConfig_.maxFakesInModule = iConfig.getParameter<uint32_t>("MaxFakesInModule");
    digiMorphingConfig_.numMorphingModules = 0;

    // the recovered pixels share the histogram with the valid ones, see FindClus
    if (digiMorphingConfig_.applyDigiMorphing and
        digiMorphingConfig_.maxFakesInModule >= pixelTopology::Phase2::maxPixInModule) {
      throw cms::Exception("Configuration")
          << "[SiPixelDigiMorphing]:"
          << " maxFakesInModule should be < " << pixelTopology::Phase2::maxPixInModule
          << " (TrackerTraits::maxPixInModule)"
          << " while " << digiMorphingConfig_.maxFakesInModule << " was provided at config level.\n";
    }
  }

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
    desc.add<bool>("DoDigiMorphing", false);
    desc.add<uint32_t>("MaxFakesInModule", ::pixelClustering::phase2Morphing::defaultMaxFakesInModule);
    // LAYER,LADDER,MODULE (coordinates can also be specified as a range FIRST-LAST where appropriate)
    desc.add<std::vector<std::string>>("barrelRegions", {"1,1-99,1-99"})->setComment("default: the first layer");
    // DISK,BLADE,SIDE,PANEL (coordinates can also be specified as a range FIRST-LAST where appropriate)
    desc.add<std::vector<std::string>>("endcapRegions", {});
    descriptions.addWithDefaultLabel(desc);
  }
  void SiPixelPhase2DigiToCluster::beginRun(edm::Run const&, edm::EventSetup const& iSetup) {
    using namespace pixelTopology;

    auto const& trackerGeometry = iSetup.getData(geomTokenBeginRun_);
    auto const& trackerTopology = iSetup.getData(topoToken_);

    auto const& dets = trackerGeometry.detUnits();

    uint32_t n_modules = 0;
    uint32_t oldLayer = std::numeric_limits<uint32_t>::max();
    uint32_t layerCount = 0;
    uint32_t bpix2Start = 0;

    // Loop over detector modules to find where BPIX2 starts
    for (auto& det : dets) {
      if (!GeomDetEnumerators::isInnerTracker(det->subDetector()))
        continue;

      DetId detId = det->geographicalId();
      auto layer = trackerTopology.layer(detId);

      if (layer != oldLayer) {
        if (layerCount == 1) {
          // layer 1 is BPIX2
          bpix2Start = n_modules;
        }
        layerCount++;
        oldLayer = layer;
      }
      n_modules++;
    }

    offsetBPIX2_ = bpix2Start;

    // select the modules where the digi morphing is applied, and record their size
    if (digiMorphingConfig_.applyDigiMorphing) {
      std::vector<std::pair<uint32_t, uint32_t>> modules;
      for (auto const* det : dets) {
        if (!GeomDetEnumerators::isInnerTracker(det->subDetector()))
          continue;
        DetId detId = det->geographicalId();
        if (siPixelMorphing::skipDetId(&trackerTopology, detId, theBarrelRegions_, theEndcapRegions_))
          continue;
        auto const& topology = dynamic_cast<PixelGeomDetUnit const&>(*det).specificTopology();
        const int nRows = topology.nrows();
        const int nCols = topology.ncolumns();
        if (nRows > static_cast<int>(::pixelClustering::phase2Morphing::maxRows) or
            nCols >= static_cast<int>(pixelTopology::Phase2::clusterBinning)) {
          throw cms::Exception("Configuration")
              << "[SiPixelDigiMorphing]: module " << detId.rawId() << " has " << nRows << " rows and " << nCols
              << " columns, while digi morphing supports up to " << ::pixelClustering::phase2Morphing::maxRows
              << " rows and " << pixelTopology::Phase2::clusterBinning - 1 << " columns\n";
        }
        modules.emplace_back(detId.rawId(), ::pixelClustering::phase2Morphing::packSize(nRows, nCols));
      }
      // sort once on the host for an efficient binary search on the device
      std::sort(modules.begin(), modules.end());

      morphingModulesHost_.clear();
      morphingModuleSizesHost_.clear();
      for (auto const& [rawId, size] : modules) {
        morphingModulesHost_.push_back(rawId);
        morphingModuleSizesHost_.push_back(size);
      }
      digiMorphingConfig_.numMorphingModules = morphingModulesHost_.size();
      morphingModulesChanged_ = true;
      edm::LogInfo("SiPixelPhase2DigiToCluster") << "digi morphing applied to " << modules.size() << " modules";
    }

    LogDebug("SiPixelPhase2DigiToCluster")
        << "beginRun: BPIX2 module start = " << offsetBPIX2_ << " (total pixel modules: " << n_modules
        << "). Offset from simplePixelTopology = " << pixelTopology::Phase2::layerStart[1] << '\n';
  }

  void SiPixelPhase2DigiToCluster::acquire(device::Event const& iEvent, device::EventSetup const& iSetup) {
    auto const& input = iEvent.get(pixelDigiToken_);

    const TrackerGeometry* geom_ = &iSetup.getData(geomToken_);

    // copy the morphing modules to the device after they have been updated
    if (morphingModulesChanged_) {
      const uint32_t n = morphingModulesHost_.size();
      morphingModulesDevice_ = cms::alpakatools::make_device_buffer<uint32_t[]>(iEvent.queue(), n);
      morphingModuleSizesDevice_ = cms::alpakatools::make_device_buffer<uint32_t[]>(iEvent.queue(), n);
      if (n > 0) {
        alpaka::memcpy(
            iEvent.queue(), *morphingModulesDevice_, cms::alpakatools::make_host_view(morphingModulesHost_.data(), n));
        alpaka::memcpy(iEvent.queue(),
                       *morphingModuleSizesDevice_,
                       cms::alpakatools::make_host_view(morphingModuleSizesHost_.data(), n));
      }
      morphingModulesChanged_ = false;
    }

    nDigis_ = 0;
    for (const auto& det : input) {
      nDigis_ += det.size();
    }
    digis_d_ = SiPixelDigisSoACollection(iEvent.queue(), nDigis_);

    if (nDigis_ == 0) {
      algo_.zeroInitializePhase2Clusters(iEvent.queue());
      return;
    }

    SiPixelDigisHost digis_h(iEvent.queue(), nDigis_);

    uint32_t nDigis = 0;
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
        ++nDigis;
      }
    }
    assert(nDigis == nDigis_);

    alpaka::memcpy(iEvent.queue(), digis_d_->buffer(), digis_h.buffer());
    algo_.makePhase2ClustersAsync(iEvent.queue(),
                                  clusterThresholds_,
                                  digis_d_->view(),
                                  nDigis_,
                                  offsetBPIX2_,
                                  digiMorphingConfig_,
                                  morphingModulesDevice_ ? morphingModulesDevice_->data() : nullptr,
                                  morphingModuleSizesDevice_ ? morphingModuleSizesDevice_->data() : nullptr);
  }

  void SiPixelPhase2DigiToCluster::produce(device::Event& iEvent, device::EventSetup const& iSetup) {
    if (nDigis_ != 0)
      digis_d_->setNModules(algo_.nModules());

    iEvent.emplace(digiPutToken_, std::move(*digis_d_));
    iEvent.emplace(clusterPutToken_, algo_.getClusters());
    digis_d_.reset();
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// define as framework plugin
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(SiPixelPhase2DigiToCluster);
