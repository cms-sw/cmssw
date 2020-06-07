#include <iostream>
#include <fstream>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/L1TObjects/interface/L1TMuonEndCapForest.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndCapForestRcd.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndCapForestO2ORcd.h"

class L1TMuonEndCapForestOnlineProxy : public edm::ESProducer {
private:
	 edm::ESGetToken<L1TMuonEndCapForest,L1TMuonEndCapForestRcd> baseSettings_token;

public:
  std::unique_ptr<L1TMuonEndCapForest> produce(const L1TMuonEndCapForestO2ORcd& record);

  L1TMuonEndCapForestOnlineProxy(const edm::ParameterSet&);
  ~L1TMuonEndCapForestOnlineProxy(void) override {}
};

L1TMuonEndCapForestOnlineProxy::L1TMuonEndCapForestOnlineProxy(const edm::ParameterSet& iConfig) : edm::ESProducer() {
  setWhatProduced(this)
  	.setConsumes(baseSettings_token);
}

std::unique_ptr<L1TMuonEndCapForest> L1TMuonEndCapForestOnlineProxy::produce(const L1TMuonEndCapForestO2ORcd& record) {
  const L1TMuonEndCapForestRcd& baseRcd = record.template getRecord<L1TMuonEndCapForestRcd>();
  auto const baseSettings = baseRcd.get(baseSettings_token);

  return std::make_unique<L1TMuonEndCapForest>(baseSettings);
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TMuonEndCapForestOnlineProxy);
