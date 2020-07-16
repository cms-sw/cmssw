#include <iostream>
#include <sstream>
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/BeamSpotPlugins/plugins/BeamSpot_PayloadInspector.cc"
#include "CondCore/BeamSpotPlugins/plugins/BeamSpotOnline_PayloadInspector.cc"
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

  // BeamSpot

  std::string tag = "BeamSpotObjects_PCL_byLumi_v0_prompt";
  cond::Time_t start = boost::lexical_cast<unsigned long long>(1406876667347162);
  //cond::Time_t end = boost::lexical_cast<unsigned long long>(1406876667347162);

  edm::LogPrint("testBeamSpotPayloadInspector") << "## Exercising BeamSpot plots " << std::endl;

  BeamSpotParameters histoParameters;
  histoParameters.process(connectionString, PI::mk_input(tag, start, start));
  edm::LogPrint("testBeamSpotPayloadInspector") << histoParameters.data() << std::endl;

  tag = "BeamSpotOnlineTestLegacy";
  start = boost::lexical_cast<unsigned long long>(1443392479297557);

  edm::LogPrint("testBeamSpotPayloadInspector") << "## Exercising BeamSpotOnline plots " << std::endl;

  BeamSpotOnlineParameters histoOnlineParameters;
  histoOnlineParameters.process(connectionString, PI::mk_input(tag, start, start));
  edm::LogPrint("testBeamSpotPayloadInspector") << histoOnlineParameters.data() << std::endl;

  Py_Finalize();
}
