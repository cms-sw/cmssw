#ifndef GTCollections_h
#define GTCollections_h

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Tau.h"

#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"

//#include "EventFilter/L1TRawToDigi/interface/UnpackerCollections.h"
#include "L1TObjectCollections.h"

namespace l1t {
  namespace stage2 {
      class GTCollections : public L1TObjectCollections {
         public:
            GTCollections(edm::Event& e) :
                 L1TObjectCollections(e),
		 egammas_(new EGammaBxCollection()),
		 etsums_(new EtSumBxCollection()),
		 jets_(new JetBxCollection()),
		 taus_(new TauBxCollection()),
		 algBlk_(new GlobalAlgBlkBxCollection()),
		 extBlk_(new GlobalExtBlkBxCollection())  {};

            virtual ~GTCollections();
            
	    inline EGammaBxCollection* getEGammas() override { return egammas_.get(); };
            inline EtSumBxCollection* getEtSums() override { return etsums_.get(); };
            inline JetBxCollection* getJets() override { return jets_.get(); };
            inline TauBxCollection* getTaus() override { return taus_.get(); };

            inline GlobalAlgBlkBxCollection* getAlgs() { return algBlk_.get(); };
            inline GlobalExtBlkBxCollection* getExts() { return extBlk_.get(); };


         private:
	    
	    std::auto_ptr<EGammaBxCollection> egammas_;
	    std::auto_ptr<EtSumBxCollection> etsums_;
	    std::auto_ptr<JetBxCollection> jets_;
	    std::auto_ptr<TauBxCollection> taus_;

            std::auto_ptr<GlobalAlgBlkBxCollection> algBlk_;
            std::auto_ptr<GlobalExtBlkBxCollection> extBlk_;


      };
   }
}

#endif
