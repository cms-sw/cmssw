#include "FWCore/Framework/interface/Event.h"

#include "BMTFCollections.h"

namespace l1t {
   namespace stage2 {
      BMTFCollections::~BMTFCollections()
      {
				event_.put(outputMuons_,"BMTF");
				//event_.put(inputMuonsPh_,"PhiDigis");
				//event_.put(inputMuonsTh_,"TheDigis");
				event_.put(inputMuonsPh_);
                                event_.put(inputMuonsTh_);

      }
   }
}
