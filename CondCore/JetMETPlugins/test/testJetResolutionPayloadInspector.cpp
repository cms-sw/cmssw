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

  std::cout << "## Jet Pt Resolution Histograms" << std::endl;

  JME::JetResolutionVsEta histo1;
  histo1.process(connectionString, PI::mk_input(tag_pt, start, end));
  std::cout << histo1.data() << std::endl;

  JME::JetResolutionVsPt histo2;
  histo2.process(connectionString, PI::mk_input(tag_pt, start, end));
  std::cout << histo2.data() << std::endl;

  std::cout << "## Jet Eta Resolution Histograms" << std::endl;

  JME::JetResolutionVsEta histo3;
  histo3.process(connectionString, PI::mk_input(tag_eta, start, end));
  std::cout << histo3.data() << std::endl;

  JME::JetResolutionVsPt histo4;
  histo4.process(connectionString, PI::mk_input(tag_eta, start, end));
  std::cout << histo4.data() << std::endl;

  std::cout << "## Jet SF Histograms" << std::endl;

  JME::JetScaleFactorVsEtaNORM histo5;
  histo5.process(connectionString, PI::mk_input(tag_sf, start, end));
  std::cout << histo5.data() << std::endl;

  JME::JetScaleFactorVsEtaDOWN histo6;
  histo6.process(connectionString, PI::mk_input(tag_sf, start, end));
  std::cout << histo6.data() << std::endl;

  JME::JetScaleFactorVsEtaUP histo7;
  histo7.process(connectionString, PI::mk_input(tag_sf, start, end));
  std::cout << histo7.data() << std::endl;

  Py_Finalize();
}
