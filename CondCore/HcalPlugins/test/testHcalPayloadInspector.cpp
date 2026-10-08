#include <iostream>
#include <sstream>
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/HcalPlugins/plugins/HcalPedestals_PayloadInspector.cc"
#include "CondCore/HcalPlugins/plugins/HcalGains_PayloadInspector.cc"
#include "CondCore/HcalPlugins/plugins/HcalPedestalWidths_PayloadInspector.cc"
#include "CondCore/HcalPlugins/plugins/HcalRespCorrs_PayloadInspector.cc"
#include "CondCore/HcalPlugins/plugins/HcalChannelQuality_PayloadInspector.cc"

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

  // ===========================================================
  // Pedestals
  // ===========================================================
  std::string tag = "HcalPedestals_ADC_v9.12_offline";
  cond::Time_t start = static_cast<unsigned long long>(1);
  cond::Time_t end = static_cast<unsigned long long>(226066);

  edm::LogPrint("testHcalPayloadInspector") << "## Exercising Pedestals plots " << std::endl;

  HcalPedestalsPlot histoPedMap;
  histoPedMap.process(connectionString, PI::mk_input(tag, start, start));
  edm::LogPrint("testHcalPayloadInspector") << histoPedMap.data() << std::endl;

  HcalPedestalsEtaPlot histoPedEta;
  histoPedEta.process(connectionString, PI::mk_input(tag, start, start));
  edm::LogPrint("testHcalPayloadInspector") << histoPedEta.data() << std::endl;

  HcalPedestalsPhiPlot histoPedPhi;
  histoPedPhi.process(connectionString, PI::mk_input(tag, start, start));
  edm::LogPrint("testHcalPayloadInspector") << histoPedPhi.data() << std::endl;

  HcalPedestalsDiff histoPedDiff;
  histoPedDiff.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testHcalPayloadInspector") << histoPedDiff.data() << std::endl;

  HcalPedestalsEtaDiff histoPedEtaDiff;
  histoPedEtaDiff.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testHcalPayloadInspector") << histoPedEtaDiff.data() << std::endl;

  HcalPedestalsPhiDiff histoPedPhiDiff;
  histoPedPhiDiff.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testHcalPayloadInspector") << histoPedPhiDiff.data() << std::endl;

  // ===========================================================
  // Gains
  // ===========================================================
  tag = "HcalGains_v11.0_offline";
  start = static_cast<unsigned long long>(368822);
  end = static_cast<unsigned long long>(381384);

  edm::LogPrint("testHcalPayloadInspector") << "## Exercising Gains plots " << std::endl;

  HcalGainsPlot histoGainMap;
  histoGainMap.process(connectionString, PI::mk_input(tag, start, start));
  edm::LogPrint("testHcalPayloadInspector") << histoGainMap.data() << std::endl;

  HcalGainsEtaPlot histoGainEta;
  histoGainEta.process(connectionString, PI::mk_input(tag, start, start));
  edm::LogPrint("testHcalPayloadInspector") << histoGainEta.data() << std::endl;

  HcalGainsPhiPlot histoGainPhi;
  histoGainPhi.process(connectionString, PI::mk_input(tag, start, start));
  edm::LogPrint("testHcalPayloadInspector") << histoGainPhi.data() << std::endl;

  HcalGainsRatio histoGainRatio;
  histoGainRatio.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testHcalPayloadInspector") << histoGainRatio.data() << std::endl;

  HcalGainsEtaRatio histoGainEtaRatio;
  histoGainEtaRatio.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testHcalPayloadInspector") << histoGainEtaRatio.data() << std::endl;

  HcalGainsPhiRatio histoGainPhiRatio;
  histoGainPhiRatio.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testHcalPayloadInspector") << histoGainPhiRatio.data() << std::endl;

  // ===========================================================
  // PedestalWidths
  // ===========================================================
  tag = "HcalPedestalWidths_ADC_v8.1_hlt";
  start = static_cast<unsigned long long>(309055);
  end = static_cast<unsigned long long>(317104);

  edm::LogPrint("testHcalPayloadInspector") << "## Exercising PedestalWidths plots " << std::endl;

  HcalPedestalWidthsPlot histoPedWMap;
  histoPedWMap.process(connectionString, PI::mk_input(tag, start, start));
  edm::LogPrint("testHcalPayloadInspector") << histoPedWMap.data() << std::endl;

  HcalPedestalWidthsEtaPlot histoPedWEta;
  histoPedWEta.process(connectionString, PI::mk_input(tag, start, start));
  edm::LogPrint("testHcalPayloadInspector") << histoPedWEta.data() << std::endl;

  HcalPedestalWidthsPhiPlot histoPedWPhi;
  histoPedWPhi.process(connectionString, PI::mk_input(tag, start, start));
  edm::LogPrint("testHcalPayloadInspector") << histoPedWPhi.data() << std::endl;

  HcalPedestalWidthsDiff histoPedWDiff;
  histoPedWDiff.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testHcalPayloadInspector") << histoPedWDiff.data() << std::endl;

  HcalPedestalWidthsEtaDiff histoPedWEtaDiff;
  histoPedWEtaDiff.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testHcalPayloadInspector") << histoPedWEtaDiff.data() << std::endl;

  HcalPedestalWidthsPhiDiff histoPedWPhiDiff;
  histoPedWPhiDiff.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testHcalPayloadInspector") << histoPedWPhiDiff.data() << std::endl;

  // ===========================================================
  // RespCorrs
  // ===========================================================
  tag = "HcalRespCorrs_v9.0_offline";
  start = static_cast<unsigned long long>(342670);
  end = static_cast<unsigned long long>(377783);

  edm::LogPrint("testHcalPayloadInspector") << "## Exercising RespCorrs plots " << std::endl;

  HcalRespCorrsPlotAll histoRespMap;
  histoRespMap.process(connectionString, PI::mk_input(tag, start, start));
  edm::LogPrint("testHcalPayloadInspector") << histoRespMap.data() << std::endl;

  HcalRespCorrsEtaPlotAll histoRespEta;
  histoRespEta.process(connectionString, PI::mk_input(tag, start, start));
  edm::LogPrint("testHcalPayloadInspector") << histoRespEta.data() << std::endl;

  HcalRespCorrsPhiPlotAll histoRespPhi;
  histoRespPhi.process(connectionString, PI::mk_input(tag, start, start));
  edm::LogPrint("testHcalPayloadInspector") << histoRespPhi.data() << std::endl;

  HcalRespCorrsRatioAll histoRespRatio;
  histoRespRatio.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testHcalPayloadInspector") << histoRespRatio.data() << std::endl;

  HcalRespCorrsEtaRatioAll histoRespEtaRatio;
  histoRespEtaRatio.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testHcalPayloadInspector") << histoRespEtaRatio.data() << std::endl;

  HcalRespCorrsPhiRatioAll histoRespPhiRatio;
  histoRespPhiRatio.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testHcalPayloadInspector") << histoRespPhiRatio.data() << std::endl;

  HcalRespCorrsPlotHBHO histoRespHBHO;
  histoRespHBHO.process(connectionString, PI::mk_input(tag, start, start));
  edm::LogPrint("testHcalPayloadInspector") << histoRespHBHO.data() << std::endl;

  HcalRespCorrsPlotHE histoRespHE;
  histoRespHE.process(connectionString, PI::mk_input(tag, start, start));
  edm::LogPrint("testHcalPayloadInspector") << histoRespHE.data() << std::endl;

  HcalRespCorrsPlotHF histoRespHF;
  histoRespHF.process(connectionString, PI::mk_input(tag, start, start));
  edm::LogPrint("testHcalPayloadInspector") << histoRespHF.data() << std::endl;

  HcalRespCorrsDepthsOverlay histoRespDepthsOverlay;
  histoRespDepthsOverlay.process(connectionString, PI::mk_input(tag, start, start));
  edm::LogPrint("testHcalPayloadInspector") << histoRespDepthsOverlay.data() << std::endl;

  HcalRespCorrsDistOverlay histoRespDistOverlay;
  histoRespDistOverlay.process(connectionString, PI::mk_input(tag, start, start));
  edm::LogPrint("testHcalPayloadInspector") << histoRespDistOverlay.data() << std::endl;

  HcalRespCorrsComparatorSingleTag histoRespComparator;
  histoRespComparator.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testHcalPayloadInspector") << histoRespComparator.data() << std::endl;

  HcalRespCorrsCorrelationSingleTag histoRespCorrelation;
  histoRespCorrelation.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testHcalPayloadInspector") << histoRespCorrelation.data() << std::endl;

  // ===========================================================
  // ChannelQuality
  // ===========================================================
  tag = "HcalChannelQuality_v7.00_offline";
  start = static_cast<unsigned long long>(272230);
  end = static_cast<unsigned long long>(273933);

  edm::LogPrint("testHcalPayloadInspector") << "## Exercising ChannelQuality plots " << std::endl;

  HcalChannelQualityPlot histoQualMap;
  histoQualMap.process(connectionString, PI::mk_input(tag, start, start));
  edm::LogPrint("testHcalPayloadInspector") << histoQualMap.data() << std::endl;

  HcalChannelQualityChange histoQualChange;
  histoQualChange.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testHcalPayloadInspector") << histoQualChange.data() << std::endl;

  HcalChannelQualityStatusChangeMap histoQualStatusChange;
  histoQualStatusChange.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testHcalPayloadInspector") << histoQualStatusChange.data() << std::endl;

  Py_Finalize();
}
