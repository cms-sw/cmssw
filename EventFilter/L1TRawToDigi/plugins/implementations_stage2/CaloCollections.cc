#include "FWCore/Framework/interface/Event.h"

#include "CaloCollections.h"

namespace l1t {
  namespace stage2 {
    CaloCollections::~CaloCollections() {
      event_.put(std::move(towers_), "CaloTower");
      event_.put(std::move(egammas_), "EGamma");
      event_.put(std::move(etsums_), "EtSum");
      event_.put(std::move(jets_), "Jet");
      event_.put(std::move(taus_), "Tau");

      event_.put(std::move(mp_etsums_), "MP");
      event_.put(std::move(mp_jets_), "MP");
      event_.put(std::move(mp_egammas_), "MP");
      event_.put(std::move(mp_taus_), "MP");
    }
  }  // namespace stage2
}  // namespace l1t
