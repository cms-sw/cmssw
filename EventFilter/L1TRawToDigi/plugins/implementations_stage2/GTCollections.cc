#include "FWCore/Framework/interface/Event.h"

#include "GTCollections.h"

namespace l1t {
   namespace stage2 {
      GTCollections::~GTCollections()
      {

        event_.put(std::move(muons_), "Muon");
	event_.put(std::move(egammas_), "EGamma");
	event_.put(std::move(etsums_), "EtSum");
	event_.put(std::move(jets_), "Jet");
	event_.put(std::move(taus_), "Tau");
	
	event_.put(std::move(algBlk_));
	event_.put(std::move(extBlk_)); 

      }
   }
}
