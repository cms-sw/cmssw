#include "FWCore/Framework/interface/Event.h"

#include "GTCollections.h"

namespace l1t {
   namespace stage2 {
      GTCollections::~GTCollections()
      {

        event_.put(muons_, "Muon");
	event_.put(egammas_, "EGamma");
	event_.put(etsums_, "EtSum");
	event_.put(jets_, "Jet");
	event_.put(taus_, "Tau");
	
	event_.put(algBlk_);
	event_.put(extBlk_); 

      }
   }
}
