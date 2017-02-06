#include "FWCore/Framework/interface/Event.h"

#include "CaloCollections.h"

namespace l1t {
   namespace stage2 {
      CaloCollections::~CaloCollections()
      {
         event_.put(towers_,"CaloTower");
         event_.put(egammas_,"EGamma");
         event_.put(etsums_,"EtSum");
         event_.put(jets_,"Jet");
         event_.put(taus_,"Tau");

         event_.put(mp_etsums_, "MP");
         event_.put(mp_jets_, "MP");
	 event_.put(mp_egammas_,"MP");
	 event_.put(mp_taus_,"MP");
      }
   }
}
