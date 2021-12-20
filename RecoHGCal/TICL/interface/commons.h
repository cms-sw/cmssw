#ifndef RecoHGCal_TICL_interface_commons_h
#define RecoHGCal_TICL_interface_commons_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"

namespace ticl {

  inline Trackster::ParticleType tracksterParticleTypeFromPdgId(int pdgId, int charge) {
    if (pdgId == 111) {
      return Trackster::ParticleType::neutral_pion;
    } else {
      pdgId = std::abs(pdgId);
      if (pdgId == 22) {
        return Trackster::ParticleType::photon;
      } else if (pdgId == 11) {
        return Trackster::ParticleType::electron;
      } else if (pdgId == 13) {
        return Trackster::ParticleType::muon;
      } else {
        bool isHadron = (pdgId > 100 and pdgId < 900) or (pdgId > 1000 and pdgId < 9000);
        if (isHadron) {
          if (charge != 0) {
            return Trackster::ParticleType::charged_hadron;
          } else {
            return Trackster::ParticleType::neutral_hadron;
          }
        } else {
          return Trackster::ParticleType::unknown;
        }
      }
    }
  }

  static void addTrackster(
      const int& index,
      const std::vector<std::pair<edm::Ref<reco::CaloClusterCollection>, std::pair<float, float>>>& lcVec,
      const std::vector<float>& inputClusterMask,
      const float& fractionCut_,
      const float& energy,
      const int& pdgId,
      const int& charge,
      const edm::ProductID& seed,
      const Trackster::IterationIndex iter,
      std::vector<float>& output_mask,
      std::vector<Trackster>& result) {
    if (lcVec.empty())
      return;

    Trackster tmpTrackster;
    tmpTrackster.zeroProbabilities();
    tmpTrackster.vertices().reserve(lcVec.size());
    tmpTrackster.vertex_multiplicity().reserve(lcVec.size());
    for (auto const& [lc, energyScorePair] : lcVec) {
      if (inputClusterMask[lc.index()] > 0) {
        double fraction = energyScorePair.first / lc->energy();
        if (fraction < fractionCut_)
          continue;
        tmpTrackster.vertices().push_back(lc.index());
        output_mask[lc.index()] -= fraction;
        tmpTrackster.vertex_multiplicity().push_back(1. / fraction);
      }
    }

    tmpTrackster.setIdProbability(tracksterParticleTypeFromPdgId(pdgId, charge), 1.f);
    tmpTrackster.setRegressedEnergy(energy);
    tmpTrackster.setIteration(iter);
    tmpTrackster.setSeed(seed, index);
    result.emplace_back(tmpTrackster);
  }

}  // namespace ticl

#endif
