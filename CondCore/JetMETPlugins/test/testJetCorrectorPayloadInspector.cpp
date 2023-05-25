#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/CondDB/interface/Time.h"

#include <iostream>
#include <sstream>
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/JetMETPlugins/plugins/JetCorrector_PayloadInspector.cc"

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

  std::string tag_mc = "JetCorrectorParametersCollection_Autumn18_V19_MC_AK4PF";
  std::string tag_data = "JetCorrectorParametersCollection_Autumn18_RunABCD_V19_DATA_AK4PF";
  cond::Time_t start = static_cast<unsigned long long>(1);
  cond::Time_t end = static_cast<unsigned long long>(1);
  py::dict inputs;

  inputs["Jet_Pt"] = "120.";
  inputs["Jet_Eta"] = "0.";
  inputs["Jet_Rho"] = "30.";

  edm::LogPrint("JEC_PI") << "## Jet Energy Corrector Vs. Eta Histograms MC" << std::endl;

  JetCorrectorVsEtaL1Offset histo1;
  histo1.setInputParamValues(inputs);
  histo1.process(connectionString, PI::mk_input(tag_mc, start, end));
  edm::LogPrint("JEC_PI") << histo1.data() << std::endl;

  JetCorrectorVsEtaL1FastJet histo2;
  histo2.setInputParamValues(inputs);
  histo2.process(connectionString, PI::mk_input(tag_mc, start, end));
  edm::LogPrint("JEC_PI") << histo2.data() << std::endl;

  JetCorrectorVsEtaL2Relative histo3;
  histo3.setInputParamValues(inputs);
  histo3.process(connectionString, PI::mk_input(tag_mc, start, end));
  edm::LogPrint("JEC_PI") << histo3.data() << std::endl;

  JetCorrectorVsEtaL3Absolute histo4;
  histo4.setInputParamValues(inputs);
  histo4.process(connectionString, PI::mk_input(tag_mc, start, end));
  edm::LogPrint("JEC_PI") << histo4.data() << std::endl;

  JetCorrectorVsEtaL2L3Residual histo5;
  histo5.setInputParamValues(inputs);
  histo5.process(connectionString, PI::mk_input(tag_mc, start, end));
  edm::LogPrint("JEC_PI") << histo5.data() << std::endl;

  JetCorrectorVsEtaUncertainty histo6;
  histo6.setInputParamValues(inputs);
  histo6.process(connectionString, PI::mk_input(tag_mc, start, end));
  edm::LogPrint("JEC_PI") << histo6.data() << std::endl;

  JetCorrectorVsEtaL1RC histo7;
  histo7.setInputParamValues(inputs);
  histo7.process(connectionString, PI::mk_input(tag_mc, start, end));
  edm::LogPrint("JEC_PI") << histo7.data() << std::endl;

  inputs.clear();
#if PY_MAJOR_VERSION >= 3
  // TODO check why this Py_INCREF is necessary...
  Py_INCREF(inputs.ptr());
#endif

  Py_Finalize();
}
