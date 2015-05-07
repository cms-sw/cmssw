#include "FWCore/Framework/interface/Event.h"

#include "GTCollections.h"

namespace l1t {
   namespace stage2 {
      GTCollections::~GTCollections()
      {

	event_.put(egammas_, "GT");
	event_.put(etsums_, "GT");
	event_.put(jets_, "GT");
	event_.put(taus_, "GT");
	
	event_.put(algBlk_);
	event_.put(extBlk_); 

      }
   }
}
