#include <iostream>
#include <sstream>
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/EcalPlugins/plugins/EcalLaserAPDPNRatios_PayloadInspector.cc"

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
  edm::OwningServiceToken servToken(edm::ServiceRegistry::createSet(psets));
  edm::ServiceRegistry::Operate operate(servToken);

  std::string connectionString("frontier://FrontierProd/CMS_CONDITIONS");

  // run-based weekly HLT tag and per-lumisection prompt tag
  std::string tag = "EcalLaserAPDPNRatios_weekly_hlt";
  std::string tagtwo = "EcalLaserAPDPNRatios_prompt_v3_inRunLSIoVs";

  cond::Time_t start = static_cast<unsigned long long>(398675);
  cond::Time_t end = static_cast<unsigned long long>(398858);
  cond::Time_t single = static_cast<unsigned long long>(398785);

  // packed run/LS IOVs of the second tag, (run << 32) | LS
  cond::Time_t startLS = static_cast<unsigned long long>(1712296086732812);
  cond::Time_t endLS = static_cast<unsigned long long>(1713082065748092);
  cond::Time_t singleLS = (static_cast<unsigned long long>(398802) << 32) | 3;

  py::dict inputs;

  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << "## Exercising single IOV plots " << std::endl;

  EcalLaserAPDPNRatiosEBMap histoEBMap;
  histoEBMap.process(connectionString, PI::mk_input(tag, single, single));
  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << histoEBMap.data() << std::endl;

  EcalLaserAPDPNRatiosEEMap histoEEMap;
  histoEEMap.process(connectionString, PI::mk_input(tag, single, single));
  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << histoEEMap.data() << std::endl;

  EcalLaserAPDPNRatiosPlot histoPlot;
  histoPlot.process(connectionString, PI::mk_input(tag, single, single));
  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << histoPlot.data() << std::endl;

  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << "## Exercising comparison plots " << std::endl;

  EcalLaserAPDPNRatiosDiffOneTag histoDiffOneTag;
  histoDiffOneTag.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << histoDiffOneTag.data() << std::endl;

  EcalLaserAPDPNRatiosDiffTwoTags histoDiffTwoTags;
  histoDiffTwoTags.process(connectionString, PI::mk_input(tag, single, single, tagtwo, singleLS, singleLS));
  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << histoDiffTwoTags.data() << std::endl;

  EcalLaserAPDPNRatiosRatioOneTag histoRatioOneTag;
  histoRatioOneTag.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << histoRatioOneTag.data() << std::endl;

  EcalLaserAPDPNRatiosRatioTwoTags histoRatioTwoTags;
  histoRatioTwoTags.process(connectionString, PI::mk_input(tag, single, single, tagtwo, singleLS, singleLS));
  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << histoRatioTwoTags.data() << std::endl;

  EcalLaserAPDPNRatiosHistCompareOneTag histoCompareOneTag;
  histoCompareOneTag.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << histoCompareOneTag.data() << std::endl;

  EcalLaserAPDPNRatiosHistCompareTwoTags histoCompareTwoTags;
  inputs["cmsLabel"] = "Preliminary";
  histoCompareTwoTags.setInputParamValues(inputs);
  histoCompareTwoTags.process(connectionString, PI::mk_input(tag, single, single, tagtwo, singleLS, singleLS));
  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << histoCompareTwoTags.data() << std::endl;
  inputs.clear();

  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << "## Exercising one tag history plots " << std::endl;

  EcalLaserAPDPNRatiosP1MeanHistoryOneTag histoP1MeanOneTag;
  histoP1MeanOneTag.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << histoP1MeanOneTag.data() << std::endl;

  EcalLaserAPDPNRatiosP2MeanHistoryOneTag histoP2MeanOneTag;
  histoP2MeanOneTag.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << histoP2MeanOneTag.data() << std::endl;

  EcalLaserAPDPNRatiosP3MeanHistoryOneTag histoP3MeanOneTag;
  histoP3MeanOneTag.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << histoP3MeanOneTag.data() << std::endl;

  EcalLaserAPDPNRatiosP1RMSHistoryOneTag histoP1RMSOneTag;
  histoP1RMSOneTag.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << histoP1RMSOneTag.data() << std::endl;

  EcalLaserAPDPNRatiosP2RMSHistoryOneTag histoP2RMSOneTag;
  histoP2RMSOneTag.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << histoP2RMSOneTag.data() << std::endl;

  EcalLaserAPDPNRatiosP3RMSHistoryOneTag histoP3RMSOneTag;
  histoP3RMSOneTag.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << histoP3RMSOneTag.data() << std::endl;

  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << "## Exercising two tags history plots " << std::endl;

  EcalLaserAPDPNRatiosP1MeanHistoryTwoTags histoP1MeanTwoTags;
  histoP1MeanTwoTags.process(connectionString, PI::mk_input(tag, start, end, tagtwo, startLS, endLS));
  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << histoP1MeanTwoTags.data() << std::endl;

  EcalLaserAPDPNRatiosP2MeanHistoryTwoTags histoP2MeanTwoTags;
  histoP2MeanTwoTags.process(connectionString, PI::mk_input(tag, start, end, tagtwo, startLS, endLS));
  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << histoP2MeanTwoTags.data() << std::endl;

  EcalLaserAPDPNRatiosP3MeanHistoryTwoTags histoP3MeanTwoTags;
  histoP3MeanTwoTags.process(connectionString, PI::mk_input(tag, start, end, tagtwo, startLS, endLS));
  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << histoP3MeanTwoTags.data() << std::endl;

  EcalLaserAPDPNRatiosP1RMSHistoryTwoTags histoP1RMSTwoTags;
  histoP1RMSTwoTags.process(connectionString, PI::mk_input(tag, start, end, tagtwo, startLS, endLS));
  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << histoP1RMSTwoTags.data() << std::endl;

  EcalLaserAPDPNRatiosP2RMSHistoryTwoTags histoP2RMSTwoTags;
  histoP2RMSTwoTags.process(connectionString, PI::mk_input(tag, start, end, tagtwo, startLS, endLS));
  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << histoP2RMSTwoTags.data() << std::endl;

  EcalLaserAPDPNRatiosP3RMSHistoryTwoTags histoP3RMSTwoTags;
  histoP3RMSTwoTags.process(connectionString, PI::mk_input(tag, start, end, tagtwo, startLS, endLS));
  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << histoP3RMSTwoTags.data() << std::endl;

  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << "## Exercising difference history plots " << std::endl;

  EcalLaserAPDPNRatiosP1DiffHistory histoP1DiffHistory;
  histoP1DiffHistory.process(connectionString, PI::mk_input(tag, start, end, tagtwo, startLS, endLS));
  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << histoP1DiffHistory.data() << std::endl;

  EcalLaserAPDPNRatiosP2DiffHistory histoP2DiffHistory;
  histoP2DiffHistory.process(connectionString, PI::mk_input(tag, start, end, tagtwo, startLS, endLS));
  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << histoP2DiffHistory.data() << std::endl;

  EcalLaserAPDPNRatiosP3DiffHistory histoP3DiffHistory;
  histoP3DiffHistory.process(connectionString, PI::mk_input(tag, start, end, tagtwo, startLS, endLS));
  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << histoP3DiffHistory.data() << std::endl;

  EcalLaserAPDPNRatiosP1PercentDiffHistory histoP1PercentDiffHistory;
  histoP1PercentDiffHistory.process(connectionString, PI::mk_input(tag, start, end, tagtwo, startLS, endLS));
  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << histoP1PercentDiffHistory.data() << std::endl;

  EcalLaserAPDPNRatiosP2PercentDiffHistory histoP2PercentDiffHistory;
  histoP2PercentDiffHistory.process(connectionString, PI::mk_input(tag, start, end, tagtwo, startLS, endLS));
  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << histoP2PercentDiffHistory.data() << std::endl;

  EcalLaserAPDPNRatiosP3PercentDiffHistory histoP3PercentDiffHistory;
  histoP3PercentDiffHistory.process(connectionString, PI::mk_input(tag, start, end, tagtwo, startLS, endLS));
  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << histoP3PercentDiffHistory.data() << std::endl;

  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << "## Exercising history plots vs time " << std::endl;

  // run based tag alone, the IOVs placed at the start time of their run from RunInfo
  EcalLaserAPDPNRatiosP2MeanHistoryOneTag histoP2MeanOneTagVsTime;
  inputs["xAxis"] = "time";
  histoP2MeanOneTagVsTime.setInputParamValues(inputs);
  histoP2MeanOneTagVsTime.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << histoP2MeanOneTagVsTime.data() << std::endl;
  inputs.clear();

  // lumi based tag alone, the lumi sections added to the start time of their run
  EcalLaserAPDPNRatiosP2MeanHistoryOneTag histoP2MeanOneTagLSVsTime;
  inputs["xAxis"] = "time";
  histoP2MeanOneTagLSVsTime.setInputParamValues(inputs);
  histoP2MeanOneTagLSVsTime.process(connectionString, PI::mk_input(tagtwo, startLS, endLS));
  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << histoP2MeanOneTagLSVsTime.data() << std::endl;
  inputs.clear();

  // run based vs lumi based tag on the common time axis
  EcalLaserAPDPNRatiosP2MeanHistoryTwoTags histoP2MeanTwoTagsVsTime;
  inputs["cmsLabel"] = "Preliminary";
  inputs["xAxis"] = "time";
  histoP2MeanTwoTagsVsTime.setInputParamValues(inputs);
  histoP2MeanTwoTagsVsTime.process(connectionString, PI::mk_input(tag, start, end, tagtwo, startLS, endLS));
  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << histoP2MeanTwoTagsVsTime.data() << std::endl;
  inputs.clear();

  EcalLaserAPDPNRatiosP2RMSHistoryTwoTags histoP2RMSTwoTagsVsTime;
  inputs["xAxis"] = "time";
  histoP2RMSTwoTagsVsTime.setInputParamValues(inputs);
  histoP2RMSTwoTagsVsTime.process(connectionString, PI::mk_input(tag, start, end, tagtwo, startLS, endLS));
  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << histoP2RMSTwoTagsVsTime.data() << std::endl;
  inputs.clear();

  EcalLaserAPDPNRatiosP2DiffHistory histoP2DiffHistoryVsTime;
  inputs["xAxis"] = "time";
  histoP2DiffHistoryVsTime.setInputParamValues(inputs);
  histoP2DiffHistoryVsTime.process(connectionString, PI::mk_input(tag, start, end, tagtwo, startLS, endLS));
  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << histoP2DiffHistoryVsTime.data() << std::endl;
  inputs.clear();

  EcalLaserAPDPNRatiosP2PercentDiffHistory histoP2PercentDiffHistoryVsTime;
  inputs["cmsLabel"] = "Preliminary";
  inputs["xAxis"] = "time";
  histoP2PercentDiffHistoryVsTime.setInputParamValues(inputs);
  histoP2PercentDiffHistoryVsTime.process(connectionString, PI::mk_input(tag, start, end, tagtwo, startLS, endLS));
  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << histoP2PercentDiffHistoryVsTime.data() << std::endl;
  inputs.clear();

  // the default run number axis requested explicitly
  EcalLaserAPDPNRatiosP2PercentDiffHistory histoP2PercentDiffHistoryVsRun;
  inputs["xAxis"] = "run";
  histoP2PercentDiffHistoryVsRun.setInputParamValues(inputs);
  histoP2PercentDiffHistoryVsRun.process(connectionString, PI::mk_input(tag, start, end, tagtwo, startLS, endLS));
  edm::LogPrint("testEcalLaserAPDPNRatiosPayloadInspector") << histoP2PercentDiffHistoryVsRun.data() << std::endl;

  inputs.clear();
#if PY_MAJOR_VERSION >= 3
  // TODO I don't know why this Py_INCREF is necessary...
  Py_INCREF(inputs.ptr());
#endif

  Py_Finalize();
}
