#ifndef CaloTokens_h
#define CaloTokens_h

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/CaloSpare.h"


#include "EventFilter/L1TRawToDigi/interface/PackerTokens.h"

namespace l1t {
   namespace stage1 {
      class CaloTokens : public PackerTokens {
         public:
            CaloTokens(const edm::ParameterSet&, edm::ConsumesCollector&);

            inline const edm::EDGetTokenT<CaloTowerBxCollection>& getCaloTowerToken() const { return towerToken_; };
            inline const edm::EDGetTokenT<EGammaBxCollection>& getEGammaToken() const { return egammaToken_; };
            inline const edm::EDGetTokenT<EtSumBxCollection>& getEtSumToken() const { return etSumToken_; };
            inline const edm::EDGetTokenT<JetBxCollection>& getJetToken() const { return jetToken_; };
            inline const edm::EDGetTokenT<TauBxCollection>& getTauToken() const { return tauToken_; };
            inline const edm::EDGetTokenT<TauBxCollection>& getIsoTauToken() const { return isotauToken_; };
            inline const edm::EDGetTokenT<CaloSpareBxCollection>& getCaloSpareHFBitCountsToken() const { return calospareHFBitCountsToken_; };
            inline const edm::EDGetTokenT<CaloSpareBxCollection>& getCaloSpareHFRingSumsToken() const { return calospareHFRingSumsToken_; };

         private:
            edm::EDGetTokenT<CaloTowerBxCollection> towerToken_;
            edm::EDGetTokenT<EGammaBxCollection> egammaToken_;
            edm::EDGetTokenT<EtSumBxCollection> etSumToken_;
            edm::EDGetTokenT<JetBxCollection> jetToken_;
            edm::EDGetTokenT<TauBxCollection> tauToken_;
            edm::EDGetTokenT<TauBxCollection> isotauToken_;
            edm::EDGetTokenT<CaloSpareBxCollection> calospareHFBitCountsToken_;
            edm::EDGetTokenT<CaloSpareBxCollection> calospareHFRingSumsToken_;
      };
   }
}

#endif
