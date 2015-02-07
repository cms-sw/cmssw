#include "FWCore/Framework/interface/Event.h"

#include "CaloCollections.h"

namespace l1t {
   namespace stage1 {
      CaloCollections::~CaloCollections()
      {
         event_.put(towers_);
         event_.put(egammas_);
         event_.put(etsums_);
         event_.put(jets_);
         event_.put(taus_, "rlxTaus");
         event_.put(isotaus_, "isoTaus");
         event_.put(calospareHFBitCounts_,"HFBitCounts");
         event_.put(calospareHFRingSums_,"HFRingSums");
         event_.put(caloEmCands_);
         event_.put(caloRegions_);
      }
   }
}
