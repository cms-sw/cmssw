#include <iostream>
#include <fstream>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "CondFormats/L1TObjects/interface/L1TMuonOverlapParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonOverlapParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1TMuonOverlapParamsO2ORcd.h"

class L1TMuonOverlapParamsOnlineProxy : public edm::ESProducer {
private:
	edm::ESGetToken<L1TMuonOverlapParams,L1TMuonOverlapParamsRcd> baseSettings_token;
public:
  std::unique_ptr<L1TMuonOverlapParams> produce(const L1TMuonOverlapParamsO2ORcd& record);

  L1TMuonOverlapParamsOnlineProxy(const edm::ParameterSet&);
  ~L1TMuonOverlapParamsOnlineProxy(void) override {}
};

L1TMuonOverlapParamsOnlineProxy::L1TMuonOverlapParamsOnlineProxy(const edm::ParameterSet& iConfig) : edm::ESProducer() {
  setWhatProduced(this)
    .setConsumes(baseSettings_token);
}

std::unique_ptr<L1TMuonOverlapParams> L1TMuonOverlapParamsOnlineProxy::produce(
    const L1TMuonOverlapParamsO2ORcd& record) {
  const L1TMuonOverlapParamsRcd& baseRcd = record.template getRecord<L1TMuonOverlapParamsRcd>();
  auto const baseSettings = baseRcd.get(baseSettings_token);

  return std::make_unique<L1TMuonOverlapParams>(baseSettings);
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1TMuonOverlapParamsOnlineProxy);
