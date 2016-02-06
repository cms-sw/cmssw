#include "FWCore/Framework/interface/Event.h"

#include "GMTCollections.h"

namespace l1t {
   namespace stage2 {
      GMTCollections::~GMTCollections()
      {
         event_.put(regionalMuonCandsBMTF_, "BMTF");
         event_.put(regionalMuonCandsOMTF_, "OMTF");
         event_.put(regionalMuonCandsEMTF_, "EMTF");
         event_.put(muons_);
      }
   }
}
