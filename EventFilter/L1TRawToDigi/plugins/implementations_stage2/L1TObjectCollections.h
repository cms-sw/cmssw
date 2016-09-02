#ifndef L1TCollections_h
#define L1TCollections_h

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Muon.h"

#include "EventFilter/L1TRawToDigi/interface/UnpackerCollections.h"

namespace l1t {
   namespace stage2 {
     class L1TObjectCollections : public UnpackerCollections {
       public:
         L1TObjectCollections(edm::Event& e) :
           UnpackerCollections(e) { };
	 virtual ~L1TObjectCollections() ;

         virtual MuonBxCollection* getMuons() { return  0;}
	 virtual EGammaBxCollection* getEGammas() { return 0;} //= 0;
	 virtual EtSumBxCollection* getEtSums() { return 0;}
	 virtual JetBxCollection* getJets() {return 0; }
	 virtual TauBxCollection* getTaus() {return 0; }
	 
      };
   }
}

#endif
