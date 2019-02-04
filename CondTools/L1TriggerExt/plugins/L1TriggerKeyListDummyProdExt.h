#ifndef CondTools_L1Trigger_L1TriggerKeyListDummyProdExt_h
#define CondTools_L1Trigger_L1TriggerKeyListDummyProdExt_h

#include <memory>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1TriggerKeyListExt.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListExtRcd.h"

class L1TriggerKeyListDummyProdExt : public edm::ESProducer {
   public:
      L1TriggerKeyListDummyProdExt(const edm::ParameterSet&);
      ~L1TriggerKeyListDummyProdExt() override;

      using ReturnType = std::unique_ptr<L1TriggerKeyListExt>;

      ReturnType produce(const L1TriggerKeyListExtRcd&);
   private:
      // ----------member data ---------------------------
};

#endif
