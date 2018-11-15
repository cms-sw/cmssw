#include "FWCore/Framework/interface/Event.h"

#include "BMTFCollections.h"

namespace l1t {
   namespace stage2 {
      BMTFCollections::~BMTFCollections()
      {
        event_.put(std::move(outputMuons_),"BMTF");
        event_.put(std::move(outputMuons2_),"BMTF2");
        //event_.put(std::move(inputMuonsPh_),"PhiDigis");
        //event_.put(std::move(inputMuonsTh_),"TheDigis");
        event_.put(std::move(inputMuonsPh_));
        event_.put(std::move(inputMuonsTh_));

      }
   }
}
