#ifndef CondTools_L1TriggerExt_L1TriggerKeyDummyProdExt_h
#define CondTools_L1TriggerExt_L1TriggerKeyDummyProdExt_h

#include <memory>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1TriggerKeyExt.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyExtRcd.h"

class L1TriggerKeyDummyProdExt : public edm::ESProducer {
   public:
      L1TriggerKeyDummyProdExt(const edm::ParameterSet&);
      ~L1TriggerKeyDummyProdExt() override;

      using ReturnType = std::unique_ptr<L1TriggerKeyExt>;

      ReturnType produce(const L1TriggerKeyExtRcd&);
   private:
      // ----------member data ---------------------------
      L1TriggerKeyExt m_key ;
};


#endif
