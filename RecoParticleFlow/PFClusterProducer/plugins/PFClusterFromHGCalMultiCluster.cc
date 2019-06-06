#include "PFClusterFromHGCalMultiCluster.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"

void PFClusterFromHGCalMultiCluster::updateEvent(const edm::Event& ev) { ev.getByToken(clusterToken_, clusterH_); }

void PFClusterFromHGCalMultiCluster::buildClusters(const edm::Handle<reco::PFRecHitCollection>& input,
                                                   const std::vector<bool>& rechitMask,
                                                   const std::vector<bool>& seedable,
                                                   reco::PFClusterCollection& output) {
  const auto& hgcalMultiClusters = *clusterH_;
  auto const& hits = *input;

  // for quick indexing back to hit energy
  std::unordered_map<uint32_t, size_t> detIdToIndex(hits.size());
  for (uint32_t i = 0; i < hits.size(); ++i) {
    detIdToIndex[hits[i].detId()] = i;
  }

  for (const auto& mcl : hgcalMultiClusters) {
    DetId seed;
    double energy = 0.0, highest_energy = 0.0;
    output.emplace_back();
    reco::PFCluster& back = output.back();
    for (const auto& cl : mcl) {
      const auto& hitsAndFractions = cl->hitsAndFractions();
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
    }    // end of loop over clusters (2D/layer)
    if (energy <= 1) {
      output.pop_back();
      continue;
    }
    if (!back.hitsAndFractions().empty()) {
      back.setSeed(seed);
      back.setEnergy(energy);
      back.setCorrectedEnergy(energy);
    } else {
      back.setSeed(0);
      back.setEnergy(0.f);
    }
  }  // end of loop over hgcalMulticlusters (3D)
}
