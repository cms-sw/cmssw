#ifndef CommonTokens_h
#define CommonTokens_h

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Muon.h"

#include "EventFilter/L1TRawToDigi/interface/PackerTokens.h"

namespace l1t {
   namespace stage2 {
      class CommonTokens : public PackerTokens {
         public:
            inline const edm::EDGetTokenT<EGammaBxCollection>& getEGammaToken() const { return egammaToken_; };
            inline const edm::EDGetTokenT<EtSumBxCollection>& getEtSumToken() const { return etSumToken_; };
            inline const edm::EDGetTokenT<JetBxCollection>& getJetToken() const { return jetToken_; };
            inline const edm::EDGetTokenT<TauBxCollection>& getTauToken() const { return tauToken_; };
	    inline const edm::EDGetTokenT<MuonBxCollection>& getMuonToken() const { return muonToken_; };

         protected:
            edm::EDGetTokenT<EGammaBxCollection> egammaToken_;
            edm::EDGetTokenT<EtSumBxCollection> etSumToken_;
            edm::EDGetTokenT<JetBxCollection> jetToken_;
            edm::EDGetTokenT<TauBxCollection> tauToken_;
            edm::EDGetTokenT<MuonBxCollection> muonToken_;
      };
   }
}

#endif
            

