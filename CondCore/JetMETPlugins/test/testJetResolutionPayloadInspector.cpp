#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/CondDB/interface/Time.h"

#include <iostream>
#include <sstream>
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/JetMETPlugins/plugins/JetResolution_PayloadInspector.cc"

#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
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

  std::string tag_pt = "JR_Fall17_V3_106X_MC_PtResolution_AK4PF";
  std::string tag_eta = "JR_Fall17_V3_106X_MC_EtaResolution_AK4PF";
  std::string tag_sf = "JR_Fall17_V3_106X_MC_SF_AK4PF";
  cond::Time_t start = static_cast<unsigned long long>(1);
  cond::Time_t end = static_cast<unsigned long long>(1);
  py::dict inputs;

  inputs["Jet_Pt"] = "120.";
  inputs["Jet_Eta"] = "0.";
  inputs["Jet_Rho"] = "30.";

  edm::LogPrint("JER_PI") << "## Jet Pt Resolution Histograms" << std::endl;

  JME::JetResolutionVsEta histo1;
  histo1.setInputParamValues(inputs);
  histo1.process(connectionString, PI::mk_input(tag_pt, start, end));
  edm::LogPrint("JER_PI") << histo1.data() << std::endl;

  JME::JetResolutionVsPt histo2;
  histo2.setInputParamValues(inputs);
  histo2.process(connectionString, PI::mk_input(tag_pt, start, end));
  edm::LogPrint("JER_PI") << histo2.data() << std::endl;

  edm::LogPrint("JER_PI") << "## Jet Eta Resolution Histograms" << std::endl;

  JME::JetResolutionVsEta histo3;
  histo3.setInputParamValues(inputs);
  histo3.process(connectionString, PI::mk_input(tag_eta, start, end));
  edm::LogPrint("JER_PI") << histo3.data() << std::endl;

  JME::JetResolutionVsPt histo4;
  histo4.setInputParamValues(inputs);
  histo4.process(connectionString, PI::mk_input(tag_eta, start, end));
  edm::LogPrint("JER_PI") << histo4.data() << std::endl;

  edm::LogPrint("JER_PI") << "## Jet SF vs. Eta Histograms" << std::endl;

  JME::JetScaleFactorVsEtaNORM histo5;
  histo5.setInputParamValues(inputs);
  histo5.process(connectionString, PI::mk_input(tag_sf, start, end));
  edm::LogPrint("JER_PI") << histo5.data() << std::endl;

  JME::JetScaleFactorVsEtaDOWN histo6;
  histo6.setInputParamValues(inputs);
  histo6.process(connectionString, PI::mk_input(tag_sf, start, end));
  edm::LogPrint("JER_PI") << histo6.data() << std::endl;

  JME::JetScaleFactorVsEtaUP histo7;
  histo7.setInputParamValues(inputs);
  histo7.process(connectionString, PI::mk_input(tag_sf, start, end));
  edm::LogPrint("JER_PI") << histo7.data() << std::endl;

  tag_sf = "JR_Autumn18_V7_MC_SF_AK4PF";

  edm::LogPrint("JER_PI") << "## Jet SF vs. Pt Histograms" << std::endl;

  JME::JetScaleFactorVsPtNORM histo8;
  histo8.setInputParamValues(inputs);
  histo8.process(connectionString, PI::mk_input(tag_sf, start, end));
  edm::LogPrint("JER_PI") << histo8.data() << std::endl;

  JME::JetScaleFactorVsPtDOWN histo9;
  histo9.setInputParamValues(inputs);
  histo9.process(connectionString, PI::mk_input(tag_sf, start, end));
  edm::LogPrint("JER_PI") << histo9.data() << std::endl;

  JME::JetScaleFactorVsPtUP histo10;
  histo10.setInputParamValues(inputs);
  histo10.process(connectionString, PI::mk_input(tag_sf, start, end));
  edm::LogPrint("JER_PI") << histo10.data() << std::endl;

  inputs.clear();
#if PY_MAJOR_VERSION >= 3
  // TODO check why this Py_INCREF is necessary...
  Py_INCREF(inputs.ptr());
#endif

  Py_Finalize();
}
