#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1TriggerKeyExtRcd.h"
#include "CondFormats/L1TObjects/interface/L1TriggerKeyExt.h"

class L1TriggerKeyExtViewer : public edm::EDAnalyzer {
private:
  std::string label;

public:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  explicit L1TriggerKeyExtViewer(const edm::ParameterSet& pset)
      : edm::EDAnalyzer(), label(pset.getParameter<std::string>("label")) {}

  ~L1TriggerKeyExtViewer(void) override {}
};

#include <iostream>
using namespace std;

void L1TriggerKeyExtViewer::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  edm::ESHandle<L1TriggerKeyExt> handle1;
  evSetup.get<L1TriggerKeyExtRcd>().get(label, handle1);
  std::shared_ptr<L1TriggerKeyExt> ptr1(new L1TriggerKeyExt(*(handle1.product())));

  cout << "L1TriggerKeyExt: parent key = " << ptr1->tscKey() << endl;

  cout << " uGT     key: " << ptr1->subsystemKey(L1TriggerKeyExt::kuGT) << endl;
  cout << " uGMT    key: " << ptr1->subsystemKey(L1TriggerKeyExt::kuGMT) << endl;
  cout << " CALO    key: " << ptr1->subsystemKey(L1TriggerKeyExt::kCALO) << endl;
  cout << " BMTF    key: " << ptr1->subsystemKey(L1TriggerKeyExt::kBMTF) << endl;
  cout << " OMTF    key: " << ptr1->subsystemKey(L1TriggerKeyExt::kOMTF) << endl;
  cout << " EMTF    key: " << ptr1->subsystemKey(L1TriggerKeyExt::kEMTF) << endl;
  cout << " TWINMUX key: " << ptr1->subsystemKey(L1TriggerKeyExt::kTWINMUX) << endl;

  cout << "Records: " << endl;

  L1TriggerKeyExt::RecordToKey::const_iterator itr = ptr1->recordToKeyMap().begin();
  L1TriggerKeyExt::RecordToKey::const_iterator end = ptr1->recordToKeyMap().end();

  for (; itr != end; ++itr) {
    std::string recordType = itr->first;
    std::string objectKey = itr->second;
    std::string recordName(recordType, 0, recordType.find_first_of("@"));
    cout << " record " << recordName << " key: " << itr->second << endl;
  }

  cout << dec << endl;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_MODULE(L1TriggerKeyExtViewer);
