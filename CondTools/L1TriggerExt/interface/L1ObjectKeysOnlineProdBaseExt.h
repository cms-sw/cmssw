#ifndef CondTools_L1TriggerExt_L1ObjectKeysOnlineProdBaseExt_h
#define CondTools_L1TriggerExt_L1ObjectKeysOnlineProdBaseExt_h

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESConsumesCollector.h"

#include "CondFormats/L1TObjects/interface/L1TriggerKeyExt.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyExtRcd.h"

#include "CondTools/L1Trigger/interface/OMDSReader.h"

// forward declarations

class L1ObjectKeysOnlineProdBaseExt : public edm::ESProducer {
public:
  L1ObjectKeysOnlineProdBaseExt(const edm::ParameterSet&);
  ~L1ObjectKeysOnlineProdBaseExt() override;

  using ReturnType = std::unique_ptr<L1TriggerKeyExt>;

  ReturnType produce(const L1TriggerKeyExtRcd&);

  virtual void fillObjectKeys(L1TriggerKeyExt* pL1TriggerKey) = 0;

private:
  // ----------member data ---------------------------
  edm::ESGetToken<L1TriggerKeyExt, L1TriggerKeyExtRcd> L1TriggerKeyExt_token;

protected:
  l1t::OMDSReader m_omdsReader;
};

#endif
