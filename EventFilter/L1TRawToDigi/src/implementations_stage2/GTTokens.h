#ifndef GTTokens_h
#define GTTokens_h

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"

#include "EventFilter/L1TRawToDigi/interface/PackerTokens.h"

namespace l1t {
  namespace stage2 {
      class GTTokens : public PackerTokens {
         public:
            GTTokens(const edm::ParameterSet&, edm::ConsumesCollector&);

            inline const edm::EDGetTokenT<EGammaBxCollection>& getEGammaToken() const { return egammaToken_; };
            inline const edm::EDGetTokenT<EtSumBxCollection>& getEtSumToken() const { return etSumToken_; };
            inline const edm::EDGetTokenT<JetBxCollection>& getJetToken() const { return jetToken_; };
            inline const edm::EDGetTokenT<TauBxCollection>& getTauToken() const { return tauToken_; };
            inline const edm::EDGetTokenT<GlobalAlgBlkBxCollection>& getAlgToken() const { return algToken_; };
            inline const edm::EDGetTokenT<GlobalExtBlkBxCollection>& getExtToken() const { return extToken_; };

         private:

	    edm::EDGetTokenT<EGammaBxCollection> egammaToken_;
	    edm::EDGetTokenT<EtSumBxCollection> etSumToken_;
	    edm::EDGetTokenT<JetBxCollection> jetToken_;
	    edm::EDGetTokenT<TauBxCollection> tauToken_;
            edm::EDGetTokenT<GlobalAlgBlkBxCollection> algToken_;
            edm::EDGetTokenT<GlobalExtBlkBxCollection> extToken_;
      };
   }
}

#endif
