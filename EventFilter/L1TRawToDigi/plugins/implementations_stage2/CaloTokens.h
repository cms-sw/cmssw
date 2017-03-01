#ifndef CaloTokens_h
#define CaloTokens_h

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Tau.h"

#include "CommonTokens.h"

namespace l1t {
   namespace stage2 {
      class CaloTokens : public CommonTokens {
         public:
            CaloTokens(const edm::ParameterSet&, edm::ConsumesCollector&);

            inline const edm::EDGetTokenT<CaloTowerBxCollection>& getCaloTowerToken() const { return towerToken_; };

         private:
            edm::EDGetTokenT<CaloTowerBxCollection> towerToken_;
      };
   }
}

#endif
