#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1TriggerKeyListExtRcd.h"
#include "CondFormats/L1TObjects/interface/L1TriggerKeyListExt.h"

class L1TriggerKeyListExtReader : public edm::EDAnalyzer {
private:
public:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  explicit L1TriggerKeyListExtReader(const edm::ParameterSet&) : edm::EDAnalyzer() {}
  ~L1TriggerKeyListExtReader(void) override {}
};

#include <iostream>
using namespace std;

void L1TriggerKeyListExtReader::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  edm::ESHandle<L1TriggerKeyListExt> handle1;
  evSetup.get<L1TriggerKeyListExtRcd>().get(handle1);
  std::shared_ptr<L1TriggerKeyListExt> ptr1(new L1TriggerKeyListExt(*(handle1.product())));

  const L1TriggerKeyListExt::KeyToToken& allKeysTokens = ptr1->tscKeyToTokenMap();
  for (auto& keyToken : allKeysTokens)
    cout << "  tscKey = " << keyToken.first << " token: " << hex << keyToken.second << dec << endl;

  const L1TriggerKeyListExt::RecordToKeyToToken& records = ptr1->recordTypeToKeyToTokenMap();
  for (auto& rec : records) {
    cout << "  " << rec.first << ":" << endl;
    for (auto& keyToken : rec.second)
      cout << "    key: " << keyToken.first << " token: " << hex << keyToken.second << dec << endl;
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_MODULE(L1TriggerKeyListExtReader);
