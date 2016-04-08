#ifndef GTTokens_h
#define GTTokens_h

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"

#include "CommonTokens.h"

namespace l1t {
  namespace stage2 {
      class GTTokens : public CommonTokens {
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
