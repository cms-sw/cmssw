#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "RecoHGCal/TICL/interface/TICLCandidateBuilderPlugins.h"

namespace ticl {
  class TracksterP4FromEnergySum final : public TracksterMomentumPluginBase {
  public:
    explicit TracksterP4FromEnergySum(const edm::ParameterSet&, edm::ConsumesCollector&& iC);
    TracksterMomentumPluginBase::LorentzVector calcP4(const ticl::Trackster& trackster) const override;
  private:
    void beginEvt() override;

    edm::InputTag vertex_src_;

    reco::Vertex vertex_;
    bool energy_from_regression_;
    edm::EDGetTokenT<std::vector<reco::Vertex> > vertex_token_;
    edm::EDGetTokenT<std::vector<reco::CaloCluster>> layer_clusters_token_;
    edm::Handle<std::vector<reco::CaloCluster>> layer_clusters_handle_;
  };

  TracksterP4FromEnergySum::TracksterP4FromEnergySum(const edm::ParameterSet& ps, edm::ConsumesCollector&& ic) : 
    TracksterMomentumPluginBase(ps, std::move(ic)),
    energy_from_regression_(ps.getParameter<bool>("energyFromRegression")),
    vertex_token_(ic.consumes<std::vector<reco::Vertex>>(ps.getParameter<edm::InputTag>("vertices"))),
    layer_clusters_token_(ic.consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("layerClusters"))) {
  }

  void TracksterP4FromEnergySum::beginEvt() {
    edm::Handle<std::vector<reco::Vertex> > vertex_h;
    evt().getByToken(vertex_token_, vertex_h);
    auto vertex_coll = *vertex_h;
    if (vertex_coll.size()) {
      const auto &vertex = vertex_coll[0];
      if (vertex.isValid() && !(vertex.isFake())) {
        vertex_ = vertex;
      }
    }

    evt().getByToken(layer_clusters_token_, layer_clusters_handle_);
  }

  TracksterP4FromEnergySum::LorentzVector TracksterP4FromEnergySum::calcP4(const ticl::Trackster& trackster) const {
    std::array<double, 3> barycentre{{0., 0., 0.}};
    double energy = 0.;
    size_t counter = 0;

    auto calo_clusters = *layer_clusters_handle_;

    for (auto idx : trackster.vertices) {
      auto fraction = 1.f / trackster.vertex_multiplicity[counter++];
      auto weight = calo_clusters[idx].energy() * fraction;
      energy += weight;
      barycentre[0] += calo_clusters[idx].x() * weight;
      barycentre[1] += calo_clusters[idx].y() * weight;
      barycentre[2] += calo_clusters[idx].z() * weight;
    }
    std::transform(
        std::begin(barycentre), std::end(barycentre), std::begin(barycentre), [&energy](double val) -> double {
          return val / energy;
        });

    math::XYZVector direction(barycentre[0] - vertex_.x(), barycentre[1] - vertex_.y(), barycentre[2] - vertex_.z());
    direction = direction.Unit();
    direction *= energy_from_regression_ ? trackster.regressed_energy : energy;

    math::XYZTLorentzVector cartesian(direction.X(), direction.Y(), direction.Z(), energy);
    // Convert px, py, pz, E vector to CMS standard pt/eta/phi/m vector
    TracksterP4FromEnergySum::LorentzVector p4(cartesian);
    return p4;
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(TracksterMomentumPluginFactory,
                  ticl::TracksterP4FromEnergySum,
                  "TracksterP4FromEnergySum");
