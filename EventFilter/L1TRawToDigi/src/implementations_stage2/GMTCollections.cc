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
         event_.put(std::move(imdMuonsBMTF_), "imdMuonsBMTF");
         event_.put(std::move(imdMuonsEMTFNeg_), "imdMuonsEMTFNeg");
         event_.put(std::move(imdMuonsEMTFPos_), "imdMuonsEMTFPos");
         event_.put(std::move(imdMuonsOMTFNeg_), "imdMuonsOMTFNeg");
         event_.put(std::move(imdMuonsOMTFPos_), "imdMuonsOMTFPos");
      }
   }
}
