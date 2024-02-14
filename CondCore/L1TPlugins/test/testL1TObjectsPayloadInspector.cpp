#include <iostream>
#include <sstream>
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/L1TPlugins/plugins/L1TUtmTriggerMenu_PayloadInspector.cc"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/PluginManager/interface/SharedLibrary.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"

int main(int argc, char** argv) {
  Py_Initialize();

  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());

  std::vector<edm::ParameterSet> psets;
  edm::ParameterSet pSet;
  pSet.addParameter("@service_type", std::string("SiteLocalConfigService"));
  psets.push_back(pSet);
  edm::ServiceToken servToken(edm::ServiceRegistry::createSet(psets));
  edm::ServiceRegistry::Operate operate(servToken);

  std::string connectionString("frontier://FrontierProd/CMS_CONDITIONS");

  // L1TUtmTriggerMenu
  std::string tag = "L1Menu_CollisionsHeavyIons2023_v1_1_5_xml";
  cond::Time_t start = static_cast<unsigned long long>(1);
  cond::Time_t end = static_cast<unsigned long long>(1);

  edm::LogPrint("testL1TObjectsPayloadInspector") << "## Exercising L1UtmTriggerMenu tests" << std::endl;

  L1TUtmTriggerMenuDisplayAlgos test1;
  test1.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testL1TObjectsPayloadInspector") << test1.data() << std::endl;

  tag = "L1TUtmTriggerMenu_Stage2v0_hlt";
  start = static_cast<unsigned long long>(375649);
  end = static_cast<unsigned long long>(375650);

  L1TUtmTriggerMenu_CompareAlgos test2;
  test2.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testL1TObjectsPayloadInspector") << test2.data() << std::endl;

  tag = "L1Menu_CollisionsHeavyIons2023_v1_1_4_xml";
  std::string tag2 = "L1Menu_CollisionsHeavyIons2023_v1_1_5_xml";
  start = static_cast<unsigned long long>(1);
  end = static_cast<unsigned long long>(1);

  L1TUtmTriggerMenu_CompareAlgosTwoTags test3;
  test3.process(connectionString, PI::mk_input(tag, start, end, tag2, start, end));
  edm::LogPrint("testL1TObjectsPayloadInspector") << test3.data() << std::endl;

  tag = "L1TGlobalPrescalesVetos_passThrough_mc";
  edm::LogPrint("testL1TObjectsPayloadInspector") << "## Exercising  L1TGlobalPrescalesVetos tests" << std::endl;

  Py_Finalize();
}
