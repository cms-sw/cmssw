#include <iomanip>
#include <iostream>

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1TUtmTriggerMenuRcd.h"
#include "CondFormats/L1TObjects/interface/L1TUtmTriggerMenu.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondCore/CondDB/interface/Session.h"

#include <iostream>
using namespace std;

class L1MenuViewer : public edm::EDAnalyzer {
public:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  explicit L1MenuViewer(const edm::ParameterSet&) : edm::EDAnalyzer() {}
  ~L1MenuViewer(void) override {}
};

void L1MenuViewer::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  edm::ESHandle<L1TUtmTriggerMenu> handle1;
  evSetup.get<L1TUtmTriggerMenuRcd>().get(handle1);
  std::shared_ptr<L1TUtmTriggerMenu> ptr1(new L1TUtmTriggerMenu(*(handle1.product())));

  cout << "L1TUtmTriggerMenu: " << endl;
  cout << " name: " << ptr1->getName() << endl;
  cout << " version: " << ptr1->getVersion() << endl;
  cout << " date/time: " << ptr1->getDatetime() << endl;
  cout << " UUID: " << ptr1->getFirmwareUuid() << endl;
  cout << " Scales: " << ptr1->getScaleSetName() << endl;
  cout << " modules: " << ptr1->getNmodules() << endl;

  cout << " Algorithms[" << ptr1->getAlgorithmMap().size() << "]: " << endl;
  for (auto a : ptr1->getAlgorithmMap())
    cout << "  " << a.first << endl;

  cout << " Conditions[" << ptr1->getConditionMap().size() << "]: " << endl;
  for (auto a : ptr1->getConditionMap())
    cout << "  " << a.first << endl;

  cout << " Conditions[" << ptr1->getScaleMap().size() << "]: " << endl;
  for (auto a : ptr1->getScaleMap())
    cout << "  " << a.first << endl;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_MODULE(L1MenuViewer);
