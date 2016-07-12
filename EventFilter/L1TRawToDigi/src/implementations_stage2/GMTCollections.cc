#include "FWCore/Framework/interface/Event.h"

#include "GMTCollections.h"

namespace l1t {
   namespace stage2 {
      GMTCollections::~GMTCollections()
      {
         event_.put(std::move(regionalMuonCandsBMTF_), "BMTF");
         event_.put(std::move(regionalMuonCandsOMTF_), "OMTF");
         event_.put(std::move(regionalMuonCandsEMTF_), "EMTF");
         event_.put(std::move(muons_), "Muon");
      }
   }
}
