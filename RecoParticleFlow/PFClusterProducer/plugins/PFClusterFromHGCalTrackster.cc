#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoParticleFlow/PFClusterProducer/interface/InitialClusteringStepBase.h"

class PFClusterFromHGCalTrackster : public InitialClusteringStepBase {
public:
  PFClusterFromHGCalTrackster(const edm::ParameterSet& conf, edm::ConsumesCollector& cc)
      : InitialClusteringStepBase(conf, cc) {
    filterByTracksterPID_ = conf.getParameter<bool>("filterByTracksterPID");
    pid_threshold_ = conf.getParameter<double>("pid_threshold");
    filter_on_categories_ = conf.getParameter<std::vector<int> >("filter_on_categories");

    tracksterToken_ = cc.consumes<std::vector<ticl::Trackster> >(conf.getParameter<edm::InputTag>("tracksterSrc"));
    clusterToken_ = cc.consumes<reco::CaloClusterCollection>(conf.getParameter<edm::InputTag>("clusterSrc"));
  }

  ~PFClusterFromHGCalTrackster() override {}
  PFClusterFromHGCalTrackster(const PFClusterFromHGCalTrackster&) = delete;
  PFClusterFromHGCalTrackster& operator=(const PFClusterFromHGCalTrackster&) = delete;

  void updateEvent(const edm::Event&) final;

  void buildClusters(const edm::Handle<reco::PFRecHitCollection>&,
                     const std::vector<bool>&,
                     const std::vector<bool>&,
                     reco::PFClusterCollection&) override;

private:
  bool filterByTracksterPID_;
  float pid_threshold_;
  std::vector<int> filter_on_categories_;

  edm::EDGetTokenT<std::vector<ticl::Trackster> > tracksterToken_;
  edm::Handle<std::vector<ticl::Trackster> > trackstersH_;

  edm::EDGetTokenT<reco::CaloClusterCollection> clusterToken_;
  edm::Handle<reco::CaloClusterCollection> clusterH_;
};

DEFINE_EDM_PLUGIN(InitialClusteringStepFactory, PFClusterFromHGCalTrackster, "PFClusterFromHGCalTrackster");

void PFClusterFromHGCalTrackster::updateEvent(const edm::Event& ev) {
  ev.getByToken(tracksterToken_, trackstersH_);
  ev.getByToken(clusterToken_, clusterH_);
}

void PFClusterFromHGCalTrackster::buildClusters(const edm::Handle<reco::PFRecHitCollection>& input,
                                                const std::vector<bool>& rechitMask,
                                                const std::vector<bool>& seedable,
                                                reco::PFClusterCollection& output) {
  auto const& hits = *input;

  const auto& tracksters = *trackstersH_;
  const auto& clusters = *clusterH_;

  // for quick indexing back to hit energy
  std::unordered_map<uint32_t, size_t> detIdToIndex(hits.size());
  for (uint32_t i = 0; i < hits.size(); ++i) {
    detIdToIndex[hits[i].detId()] = i;
  }

  for (const auto& tst : tracksters) {
    // Skip empty tracksters
    if (tst.vertices().empty()) {
      continue;
    }
    // Filter using trackster PID
    if (filterByTracksterPID_) {
      float probTotal = 0.0f;
      for (int cat : filter_on_categories_) {
        probTotal += tst.id_probabilities(cat);
      }
      if (probTotal < pid_threshold_) {
        continue;
      }
    }

    DetId seed;
    double energy = 0.0, highest_energy = 0.0;
    output.emplace_back();
    reco::PFCluster& back = output.back();

    std::vector<std::pair<DetId, float> > hitsAndFractions;
    int iLC = 0;
    std::for_each(std::begin(tst.vertices()), std::end(tst.vertices()), [&](unsigned int lcId) {
      const auto fraction = 1.f / tst.vertex_multiplicity(iLC++);
      for (const auto& cell : clusters[lcId].hitsAndFractions()) {
        hitsAndFractions.emplace_back(cell.first, cell.second * fraction);
      }
    });

    for (const auto& hAndF : hitsAndFractions) {
      auto itr = detIdToIndex.find(hAndF.first);
      if (itr == detIdToIndex.end()) {
        continue;  // hit wasn't saved in reco
      }
      auto ref = makeRefhit(input, itr->second);
      assert(ref->detId() == hAndF.first.rawId());
      const double hit_energy = hAndF.second * ref->energy();
      energy += hit_energy;
      back.addRecHitFraction(reco::PFRecHitFraction(ref, hAndF.second));
      // TODO: the following logic to identify the seed of a cluster
      // could be appropriate for the Run2 Ecal Calorimetric
      // detector, but could be wrong for the HGCal one. This has to
      // be reviewd.
      if (hit_energy > highest_energy || highest_energy == 0.0) {
        highest_energy = hit_energy;
        seed = ref->detId();
      }
    }  // end of hitsAndFractions

    if (!back.hitsAndFractions().empty()) {
      back.setSeed(seed);
      back.setEnergy(energy);
      back.setCorrectedEnergy(energy);
    } else {
      back.setSeed(0);
      back.setEnergy(0.f);
    }
  }  // end of loop over hgcalTracksters (3D)
}
