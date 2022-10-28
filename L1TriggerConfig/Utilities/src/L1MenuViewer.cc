#include <iomanip>
#include <iostream>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "CondFormats/DataRecord/interface/L1TUtmTriggerMenuRcd.h"
#include "CondFormats/L1TObjects/interface/L1TUtmTriggerMenu.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondCore/CondDB/interface/Session.h"

using namespace std;

class L1MenuViewer : public edm::one::EDAnalyzer<> {
public:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  const edm::ESGetToken<L1TUtmTriggerMenu, L1TUtmTriggerMenuRcd> l1GtMenuToken_;

  explicit L1MenuViewer(const edm::ParameterSet&)
      : edm::one::EDAnalyzer<>(), l1GtMenuToken_(esConsumes<L1TUtmTriggerMenu, L1TUtmTriggerMenuRcd>()) {}
  ~L1MenuViewer(void) override = default;
};

void L1MenuViewer::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  auto const& l1GtMenu = evSetup.getData(l1GtMenuToken_);

  cout << "L1TUtmTriggerMenu: " << endl;
  cout << " name: " << l1GtMenu.getName() << endl;
  cout << " version: " << l1GtMenu.getVersion() << endl;
  cout << " date/time: " << l1GtMenu.getDatetime() << endl;
  cout << " UUID: " << l1GtMenu.getFirmwareUuid() << endl;
  cout << " Scales: " << l1GtMenu.getScaleSetName() << endl;
  cout << " modules: " << l1GtMenu.getNmodules() << endl;

  cout << " Algorithms[" << l1GtMenu.getAlgorithmMap().size() << "]: " << endl;
  for (const auto& a : l1GtMenu.getAlgorithmMap())
    cout << "  " << a.first << endl;

  cout << " Conditions[" << l1GtMenu.getConditionMap().size() << "]: " << endl;
  for (const auto& a : l1GtMenu.getConditionMap())
    cout << "  " << a.first << endl;

  cout << " Conditions[" << l1GtMenu.getScaleMap().size() << "]: " << endl;
  for (const auto& a : l1GtMenu.getScaleMap())
    cout << "  " << a.first << endl;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_MODULE(L1MenuViewer);
