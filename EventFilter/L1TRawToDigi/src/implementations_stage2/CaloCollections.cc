#include "FWCore/Framework/interface/Event.h"

#include "CaloCollections.h"

namespace l1t {
   namespace stage2 {
      CaloCollections::~CaloCollections()
      {
         event_.put(towers_);
         event_.put(egammas_);
         event_.put(etsums_);
         event_.put(jets_);
         event_.put(taus_);

         event_.put(mp_etsums_, "MP");
         event_.put(mp_jets_, "MP");
      }
   }
}
