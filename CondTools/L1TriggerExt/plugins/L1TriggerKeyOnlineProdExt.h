#ifndef CondTools_L1TriggerExt_L1TriggerKeyOnlineProdExt_h
#define CondTools_L1TriggerExt_L1TriggerKeyOnlineProdExt_h

#include <memory>
#include <vector>
#include <string>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1TriggerKeyExt.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyExtRcd.h"

class L1TriggerKeyOnlineProdExt : public edm::ESProducer {
   public:
      L1TriggerKeyOnlineProdExt(const edm::ParameterSet&);
      ~L1TriggerKeyOnlineProdExt() override;

      using ReturnType = std::unique_ptr<L1TriggerKeyExt>;

      ReturnType produce(const L1TriggerKeyExtRcd&);
   private:
      // ----------member data ---------------------------
      std::vector< std::string > m_subsystemLabels ;
};

#endif
