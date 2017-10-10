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
	 ~L1TObjectCollections() override ;

         virtual MuonBxCollection* getMuons(const unsigned int copy) { return  nullptr;}
	 virtual EGammaBxCollection* getEGammas() { return nullptr;} //= 0;
	 virtual EtSumBxCollection* getEtSums() { return nullptr;}
	 virtual JetBxCollection* getJets() {return nullptr; }
	 virtual TauBxCollection* getTaus() {return nullptr; }
	 
      };
   }
}

#endif
