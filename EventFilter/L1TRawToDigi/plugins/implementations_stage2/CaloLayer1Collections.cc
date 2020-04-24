#include "FWCore/Framework/interface/Event.h"

#include "CaloLayer1Collections.h"

namespace l1t {
   namespace stage2 {
      CaloLayer1Collections::~CaloLayer1Collections()
      {
         event_.put(std::move(ecalDigis_));
         event_.put(std::move(hcalDigis_));
         event_.put(std::move(caloRegions_));
      }
   }
}
