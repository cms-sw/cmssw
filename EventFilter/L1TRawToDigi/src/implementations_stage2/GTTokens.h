#ifndef GTTokens_h
#define GTTokens_h

#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"

#include "EventFilter/L1TRawToDigi/interface/PackerTokens.h"

namespace l1t {
  namespace stage2 {
      class GTTokens : public PackerTokens {
         public:
            GTTokens(const edm::ParameterSet&, edm::ConsumesCollector&);

            inline const edm::EDGetTokenT<GlobalAlgBlkBxCollection>& getAlgToken() const { return algToken_; };
            inline const edm::EDGetTokenT<GlobalExtBlkBxCollection>& getExtToken() const { return extToken_; };

         private:

            edm::EDGetTokenT<GlobalAlgBlkBxCollection> algToken_;
            edm::EDGetTokenT<GlobalExtBlkBxCollection> extToken_;
      };
   }
}

#endif
