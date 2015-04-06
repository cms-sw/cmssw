#ifndef GTCollections_h
#define GTCollections_h

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Tau.h"

#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"

#include "EventFilter/L1TRawToDigi/interface/UnpackerCollections.h"

namespace l1t {
  namespace stage2 {
      class GTCollections : public UnpackerCollections {
         public:
            GTCollections(edm::Event& e) :
               UnpackerCollections(e),
		 
		 algBlk_(new GlobalAlgBlkBxCollection()),
		 extBlk_(new GlobalExtBlkBxCollection())  {};

            virtual ~GTCollections();
            
            inline GlobalAlgBlkBxCollection* getAlgs() { return algBlk_.get(); };
            inline GlobalExtBlkBxCollection* getExts() { return extBlk_.get(); };


         private:
	    
            std::auto_ptr<GlobalAlgBlkBxCollection> algBlk_;
            std::auto_ptr<GlobalExtBlkBxCollection> extBlk_;


      };
   }
}

#endif
