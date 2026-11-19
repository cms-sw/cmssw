#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "CondFormats/HGCalObjects/interface/TICLGeomHost.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoARecHitsDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoARecHitsExtraDeviceCollection.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "HGCalCLUEsteringAlgoWrapper.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Clusters the rechits of one HGCAL component, layer by layer, with CLUEstering: produces the layer cluster and the
  // seed flag of each rechit, and the number of layer clusters
  class HGCalCLUEsteringLayerClustersProducer : public stream::EDProducer<> {
  public:
    HGCalCLUEsteringLayerClustersProducer(const edm::ParameterSet& config)
        : EDProducer(config),
          recHitsToken_{consumes(config.getParameter<edm::InputTag>("hgcalRecHitsSoA"))},
          layerSizesToken_{consumes<std::vector<uint32_t>>(
              edm::InputTag(config.getParameter<edm::InputTag>("hgcalRecHitsSoA").label(), "layerSizes"))},
          recHitsClusteringToken_{produces()},
          detector_(config.getParameter<std::string>("detector")),
          algo_{makeAlgo(config)} {
      if (config.getParameter<bool>("extremesFromGeometry")) {
        geometryToken_ = esConsumes();
      }
    }

    void produce(device::Event& iEvent, const device::EventSetup& iSetup) override {
      if (geometryToken_.isInitialized() and geometryWatcher_.check(iSetup)) {
        setExtremes(iSetup.getData(geometryToken_));
      }

      const auto& recHits = iEvent.get(recHitsToken_);
      // The rechits are grouped by layer, and layerSizes holds the number of rechits of each layer that has some
      const auto& layerSizes = iEvent.get(layerSizesToken_);
      const uint32_t nRecHits = recHits->metadata().size();
      if (std::accumulate(layerSizes.begin(), layerSizes.end(), uint64_t{0}) != nRecHits) {
        throw edm::Exception(edm::errors::LogicError)
            << "HGCalCLUEsteringLayerClustersProducer: the layer sizes do not add up to the " << nRecHits
            << " rechits.";
      }
      // One row per rechit, like the rechit SoA
      HGCalSoARecHitsExtraDeviceCollection recHitsClustering(iEvent.queue(), nRecHits);

      algo_.run(
          iEvent.queue(), nRecHits, std::span<const uint32_t>(layerSizes), recHits.view(), recHitsClustering.view());
      iEvent.emplace(recHitsClusteringToken_, std::move(recHitsClustering));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("hgcalRecHitsSoA", edm::InputTag("TO BE DEFINED"));
      desc.ifValue(edm::ParameterDescription<std::string>(
                       "detector",
                       "EE",
                       true,
                       edm::Comment("HGCAL component; 'BH' selects the periodic (eta,phi) scintillator metric.")),
                   edm::allowedValues<std::string>("EE", "FH", "BH", "HFNose"));
      desc.add<float>("deltac", 1.3);
      desc.add<float>("kappa", 9.);
      desc.add<float>("outlierDeltaFactor", 2.);
      desc.add<bool>("extremesFromGeometry", false)
          ->setComment(
              "Take the ranges of the coordinates from the TICLGeom cells of the component, not from the rechits of "
              "every event.");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    // CLUEstering divides by deltac and kappa, and needs the outlier distance at least the density radius
    static HGCalCLUEsteringAlgoWrapper makeAlgo(const edm::ParameterSet& config) {
      const float deltac = config.getParameter<float>("deltac");
      const float kappa = config.getParameter<float>("kappa");
      const float outlierDeltaFactor = config.getParameter<float>("outlierDeltaFactor");
      if (not(deltac > 0.f) or not(kappa > 0.f)) {
        throw edm::Exception(edm::errors::Configuration) << "HGCalCLUEsteringLayerClustersProducer: 'deltac' ("
                                                         << deltac << ") and 'kappa' (" << kappa << ") must be > 0.";
      }
      if (not(outlierDeltaFactor >= 1.f)) {
        throw edm::Exception(edm::errors::Configuration)
            << "HGCalCLUEsteringLayerClustersProducer: 'outlierDeltaFactor' (" << outlierDeltaFactor
            << ") must be >= 1.";
      }
      return HGCalCLUEsteringAlgoWrapper(
          deltac, kappa, outlierDeltaFactor, config.getParameter<std::string>("detector") == "BH");
    }

    // Sets the ranges of the coordinates to the ones of the cells of the component: (x, y) for silicon, (eta, phi)
    // for the scintillator. For HFNose, whose cells are not in TICLGeom, CLUEstering computes them in every event.
    void setExtremes(const TICLGeomHost& geometry) {
      uint32_t det = 0;
      if (detector_ == "EE") {
        det = DetId::HGCalEE;
      } else if (detector_ == "FH") {
        det = DetId::HGCalHSi;
      } else if (detector_ == "BH") {
        det = DetId::HGCalHSc;
      } else {
        return;
      }
      std::pair<float, float> dim1{std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest()};
      std::pair<float, float> dim2 = dim1;
      const auto cells = geometry.const_view().common();
      for (int32_t i = 0; i < cells.metadata().size(); ++i) {
        if (DetId(cells[i].rawDetId()).det() != det) {
          continue;
        }
        const float x = cells[i].x();
        const float y = cells[i].y();
        const float c1 = det == DetId::HGCalHSc ? std::asinh(cells[i].z() / std::hypot(x, y)) : x;
        const float c2 = det == DetId::HGCalHSc ? 0.f : y;
        dim1 = {std::min(dim1.first, c1), std::max(dim1.second, c1)};
        dim2 = {std::min(dim2.first, c2), std::max(dim2.second, c2)};
      }
      if (det == DetId::HGCalHSc) {
        // phi is periodic: CLUEstering takes its range from the metric
        dim2 = {0.f, 2.f * static_cast<float>(M_PI)};
      }
      if (dim1.first < dim1.second and dim2.first < dim2.second) {
        algo_.setExtremes(dim1, dim2);
      }
    }

    const device::EDGetToken<HGCalSoARecHitsDeviceCollection> recHitsToken_;
    const edm::EDGetTokenT<std::vector<uint32_t>> layerSizesToken_;
    const device::EDPutToken<HGCalSoARecHitsExtraDeviceCollection> recHitsClusteringToken_;
    edm::ESGetToken<TICLGeomHost, CaloGeometryRecord> geometryToken_;
    const std::string detector_;
    edm::ESWatcher<CaloGeometryRecord> geometryWatcher_;
    HGCalCLUEsteringAlgoWrapper algo_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(HGCalCLUEsteringLayerClustersProducer);
