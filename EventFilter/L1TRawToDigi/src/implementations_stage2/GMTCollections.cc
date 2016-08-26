#include "FWCore/Framework/interface/Event.h"

#include "GMTCollections.h"

namespace l1t {
   namespace stage2 {
      GMTCollections::~GMTCollections()
      {
         event_.put(regionalMuonCandsBMTF_, "BMTF");
         event_.put(regionalMuonCandsOMTF_, "OMTF");
         event_.put(regionalMuonCandsEMTF_, "EMTF");
         event_.put(muons_, "Muon");
         event_.put(imdMuonsBMTF_, "imdMuonsBMTF");
         event_.put(imdMuonsEMTFNeg_, "imdMuonsEMTFNeg");
         event_.put(imdMuonsEMTFPos_, "imdMuonsEMTFPos");
         event_.put(imdMuonsOMTFNeg_, "imdMuonsOMTFNeg");
         event_.put(imdMuonsOMTFPos_, "imdMuonsOMTFPos");
      }
   }
}
