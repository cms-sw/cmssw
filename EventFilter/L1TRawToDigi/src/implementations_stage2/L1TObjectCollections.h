#ifndef L1TCollections_h
#define L1TCollections_h

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Tau.h"

#include "EventFilter/L1TRawToDigi/interface/UnpackerCollections.h"

namespace l1t {
   namespace stage2 {
     class L1TObjectCollections : public UnpackerCollections {
       public:
         L1TObjectCollections(edm::Event& e) :
           UnpackerCollections(e) { };
	 virtual ~L1TObjectCollections() ;

	 virtual EGammaBxCollection* getEGammas() = 0;
	 virtual EtSumBxCollection* getEtSums() = 0;
	 virtual JetBxCollection* getJets() = 0;
	 virtual TauBxCollection* getTaus() = 0;
	 
      };
   }
}

#endif
