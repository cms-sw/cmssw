#include "FWCore/Framework/interface/Event.h"

#include "GTCollections.h"

namespace l1t {
  namespace stage2 {
    GTCollections::~GTCollections() {
      event_.put(std::move(muons_[0]), "Muon");
      event_.put(std::move(muonShowers_[0]), "MuonShower");
      event_.put(std::move(egammas_[0]), "EGamma");
      event_.put(std::move(etsums_[0]), "EtSum");
      event_.put(std::move(zdcsums_[0]), "EtSumZDC");
      event_.put(std::move(jets_[0]), "Jet");
      event_.put(std::move(taus_[0]), "Tau");
      for (int i = 1; i < 6; ++i) {
        event_.put(std::move(muons_[i]), "Muon" + std::to_string(i + 1));
        event_.put(std::move(muonShowers_[i]), "MuonShower" + std::to_string(i + 1));
        event_.put(std::move(egammas_[i]), "EGamma" + std::to_string(i + 1));
        event_.put(std::move(etsums_[i]), "EtSum" + std::to_string(i + 1));
        event_.put(std::move(zdcsums_[i]), "EtSumZDC" + std::to_string(i + 1));
        event_.put(std::move(jets_[i]), "Jet" + std::to_string(i + 1));
        event_.put(std::move(taus_[i]), "Tau" + std::to_string(i + 1));
      }

      event_.put(std::move(algBlk_));
      event_.put(std::move(extBlk_));
    }
  }  // namespace stage2
}  // namespace l1t
