// Plugin for getting the four-vector from a Trackster from a simple energy sum and weighted cluster position.
// A simplistic 1/N(tracksters) sharing is applied for hits that belong to multiple tracksters.
// Alternatively takes the energy value from the pre-calculated regressed energy value in the Trackster.

#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "RecoHGCal/TICL/interface/TracksterMomentumPluginBase.h"

namespace ticl {
  class TracksterP4FromEnergySum final : public TracksterMomentumPluginBase {
  public:
    explicit TracksterP4FromEnergySum(const edm::ParameterSet&, edm::ConsumesCollector&& iC);
    void setP4(const std::vector<const Trackster*>& tracksters,
               std::vector<TICLCandidate>& ticl_cands,
               edm::Event& event) const override;

  private:
    std::tuple<TracksterMomentumPluginBase::LorentzVector, float> calcP4(
        const ticl::Trackster& trackster,
        const reco::Vertex& vertex,
        const std::vector<reco::CaloCluster>& calo_clusters) const;
    bool energy_from_regression_;
    edm::EDGetTokenT<std::vector<reco::Vertex>> vertex_token_;
    edm::EDGetTokenT<std::vector<reco::CaloCluster>> layer_clusters_token_;
  };

  TracksterP4FromEnergySum::TracksterP4FromEnergySum(const edm::ParameterSet& ps, edm::ConsumesCollector&& ic)
      : TracksterMomentumPluginBase(ps, std::move(ic)),
        energy_from_regression_(ps.getParameter<bool>("energyFromRegression")),
        vertex_token_(ic.consumes<std::vector<reco::Vertex>>(ps.getParameter<edm::InputTag>("vertices"))),
        layer_clusters_token_(
            ic.consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("layerClusters"))) {}

  void TracksterP4FromEnergySum::setP4(const std::vector<const Trackster*>& tracksters,
                                       std::vector<TICLCandidate>& ticl_cands,
                                       edm::Event& event) const {
    // Find best vertex
    edm::Handle<std::vector<reco::Vertex>> vertex_h;
    event.getByToken(vertex_token_, vertex_h);
    auto vertex_coll = *vertex_h;
    reco::Vertex best_vertex;
    if (!vertex_coll.empty()) {
      const auto& vertex = vertex_coll[0];
      if (vertex.isValid() && !(vertex.isFake())) {
        best_vertex = vertex;
      }
    }

    edm::Handle<std::vector<reco::CaloCluster>> layer_clusters_h;
    event.getByToken(layer_clusters_token_, layer_clusters_h);

    auto size = std::min(tracksters.size(), ticl_cands.size());
    for (size_t i = 0; i < size; ++i) {
      const auto* trackster = tracksters[i];
      auto ret = calcP4(*trackster, best_vertex, *layer_clusters_h);

      auto& ticl_cand = ticl_cands[i];
      ticl_cand.setP4(std::get<0>(ret));
      ticl_cand.setRawEnergy(std::get<1>(ret));
    }
  }

  std::tuple<TracksterMomentumPluginBase::LorentzVector, float> TracksterP4FromEnergySum::calcP4(
      const ticl::Trackster& trackster,
      const reco::Vertex& vertex,
      const std::vector<reco::CaloCluster>& calo_clusters) const {
    std::array<double, 3> barycentre{{0., 0., 0.}};
    double energy = 0.;
    size_t counter = 0;

    for (auto idx : trackster.vertices) {
      auto n_vertices = trackster.vertex_multiplicity[counter++];
      auto fraction = n_vertices ? 1.f / n_vertices : 1.f;
      auto weight = calo_clusters[idx].energy() * fraction;
      energy += weight;
      barycentre[0] += calo_clusters[idx].x() * weight;
      barycentre[1] += calo_clusters[idx].y() * weight;
      barycentre[2] += calo_clusters[idx].z() * weight;
    }
    std::transform(
        std::begin(barycentre), std::end(barycentre), std::begin(barycentre), [&energy](double val) -> double {
          return energy >= 0. ? val / energy : val;
        });

    math::XYZVector direction(barycentre[0] - vertex.x(), barycentre[1] - vertex.y(), barycentre[2] - vertex.z());
    direction = direction.Unit();
    auto raw_energy = energy;
    energy = energy_from_regression_ ? trackster.regressed_energy : raw_energy;
    direction *= energy;

    math::XYZTLorentzVector cartesian(direction.X(), direction.Y(), direction.Z(), energy);
    // Convert px, py, pz, E vector to CMS standard pt/eta/phi/m vector
    TracksterP4FromEnergySum::LorentzVector p4(cartesian);
    return std::tuple(p4, raw_energy);
  }
}  // namespace ticl

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(TracksterMomentumPluginFactory, ticl::TracksterP4FromEnergySum, "TracksterP4FromEnergySum");
