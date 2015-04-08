#include "FWCore/Framework/interface/Event.h"

#include "GTCollections.h"

namespace l1t {
   namespace stage2 {
      GTCollections::~GTCollections()
      {
	
         event_.put(algBlk_);
         event_.put(extBlk_); 


      }
   }
}
