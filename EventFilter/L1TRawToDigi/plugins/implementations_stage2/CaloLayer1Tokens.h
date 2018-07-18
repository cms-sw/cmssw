#ifndef CaloLayer1Tokens_h
#define CaloLayer1Tokens_h

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "EventFilter/L1TRawToDigi/interface/PackerTokens.h"

namespace l1t {
   namespace stage2 {
      class CaloLayer1Tokens : public PackerTokens {
         public:
            CaloLayer1Tokens(const edm::ParameterSet&, edm::ConsumesCollector&);

            inline const edm::EDGetTokenT<EcalTrigPrimDigiCollection>& getEcalDigiToken() const {return ecalDigiToken_;};
            inline const edm::EDGetTokenT<HcalTrigPrimDigiCollection>& getHcalDigiToken() const {return hcalDigiToken_;};
            inline const edm::EDGetTokenT<L1CaloRegionCollection>&     getCaloRegionToken() const {return caloRegionToken_;};

         private:
            edm::EDGetTokenT<EcalTrigPrimDigiCollection> ecalDigiToken_;
            edm::EDGetTokenT<HcalTrigPrimDigiCollection> hcalDigiToken_;
            edm::EDGetTokenT<L1CaloRegionCollection>     caloRegionToken_;

      };
   }
}

#endif
