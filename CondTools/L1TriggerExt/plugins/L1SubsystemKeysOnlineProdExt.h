#ifndef CondTools_L1TriggerExt_L1SubsystemKeysOnlineProdExt_h
#define CondTools_L1TriggerExt_L1SubsystemKeysOnlineProdExt_h

#include <memory>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1TriggerKeyExt.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyExtRcd.h"

#include "CondTools/L1Trigger/interface/OMDSReader.h"

class L1SubsystemKeysOnlineProdExt : public edm::ESProducer {
   public:
      L1SubsystemKeysOnlineProdExt(const edm::ParameterSet&);
      ~L1SubsystemKeysOnlineProdExt() override;

      using ReturnType = std::unique_ptr<L1TriggerKeyExt>;

      ReturnType produce(const L1TriggerKeyExtRcd&);
   private:
      // ----------member data ---------------------------
      std::string m_tscKey, m_rsKey ;
      l1t::OMDSReader m_omdsReader ;
      bool m_forceGeneration ;
};

#endif
