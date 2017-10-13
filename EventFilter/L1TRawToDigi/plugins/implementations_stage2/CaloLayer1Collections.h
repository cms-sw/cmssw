#ifndef CaloLayer1Collections_h
#define CaloLayer1Collections_h


#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "EventFilter/L1TRawToDigi/interface/UnpackerCollections.h"

namespace l1t {
   namespace stage2 {
     class CaloLayer1Collections : public UnpackerCollections {
         public:
            CaloLayer1Collections(edm::Event& e) :
               UnpackerCollections(e),
               ecalDigis_(new EcalTrigPrimDigiCollection()),
               hcalDigis_(new HcalTrigPrimDigiCollection()),
               caloRegions_(new L1CaloRegionCollection())
            {};

            ~CaloLayer1Collections() override;

            inline EcalTrigPrimDigiCollection* getEcalDigis() {return ecalDigis_.get();};
            inline HcalTrigPrimDigiCollection* getHcalDigis() { return hcalDigis_.get(); };
            inline L1CaloRegionCollection* getRegions() { return caloRegions_.get(); };

         private:
            std::unique_ptr<EcalTrigPrimDigiCollection> ecalDigis_;
            std::unique_ptr<HcalTrigPrimDigiCollection> hcalDigis_;
            std::unique_ptr<L1CaloRegionCollection>     caloRegions_;
      };
   }
}

#endif
