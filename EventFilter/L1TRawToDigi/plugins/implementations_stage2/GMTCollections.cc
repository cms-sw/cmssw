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
         event_.put(std::move(muonsSet2_), "MuonSet2");
         event_.put(std::move(muonsSet3_), "MuonSet3");
         event_.put(std::move(muonsSet4_), "MuonSet4");
         event_.put(std::move(muonsSet5_), "MuonSet5");
         event_.put(std::move(muonsSet6_), "MuonSet6");
         event_.put(std::move(imdMuonsBMTF_), "imdMuonsBMTF");
         event_.put(std::move(imdMuonsEMTFNeg_), "imdMuonsEMTFNeg");
         event_.put(std::move(imdMuonsEMTFPos_), "imdMuonsEMTFPos");
         event_.put(std::move(imdMuonsOMTFNeg_), "imdMuonsOMTFNeg");
         event_.put(std::move(imdMuonsOMTFPos_), "imdMuonsOMTFPos");
      }

      MuonBxCollection*
      GMTCollections::getMuons(const unsigned int set)
      {
         if (set == 1) return muons_.get();
         else if (set == 2) return muonsSet2_.get();
         else if (set == 3) return muonsSet3_.get();
         else if (set == 4) return muonsSet4_.get();
         else if (set == 5) return muonsSet5_.get();
         else if (set == 6) return muonsSet6_.get();
         return muons_.get();
      }
   }
}
