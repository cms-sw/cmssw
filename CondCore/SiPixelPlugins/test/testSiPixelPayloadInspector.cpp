#include <iostream>
#include <sstream>
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/SiPixelPlugins/plugins/SiPixelLorentzAngle_PayloadInspector.cc"
#include "CondCore/SiPixelPlugins/plugins/SiPixelQuality_PayloadInspector.cc"
#include "CondCore/SiPixelPlugins/plugins/SiPixelGainCalibrationOffline_PayloadInspector.cc"
#include "CondCore/SiPixelPlugins/plugins/SiPixelTemplateDBObject_PayloadInspector.cc"
#include "CondCore/SiPixelPlugins/plugins/SiPixelGenErrorDBObject_PayloadInspector.cc"
#include "CondCore/SiPixelPlugins/plugins/SiPixelVCal_PayloadInspector.cc"
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

  // Lorentz Angle

  std::string tag = "SiPixelLorentzAngle_v11_offline";
  cond::Time_t start = boost::lexical_cast<unsigned long long>(303790);
  cond::Time_t end = boost::lexical_cast<unsigned long long>(324245);

  edm::LogPrint("testSiPixelPayloadInspector") << "## Exercising Lorentz Angle plots " << std::endl;

  SiPixelLorentzAngleValues histo1;
  histo1.process(connectionString, PI::mk_input(tag, start, start));
  edm::LogPrint("testSiPixelPayloadInspector") << histo1.data() << std::endl;

  SiPixelLorentzAngleValueComparisonSingleTag histo2;
  histo2.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testSiPixelPayloadInspector") << histo2.data() << std::endl;

  SiPixelLorentzAngleByRegionComparisonSingleTag histo3;
  histo3.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testSiPixelPayloadInspector") << histo3.data() << std::endl;

  SiPixelBPixLorentzAngleMap histo4;
  histo4.process(connectionString, PI::mk_input(tag, start, start));
  edm::LogPrint("testSiPixelPayloadInspector") << histo4.data() << std::endl;

  SiPixelFPixLorentzAngleMap histo5;
  histo5.process(connectionString, PI::mk_input(tag, end, end));
  edm::LogPrint("testSiPixelPayloadInspector") << histo5.data() << std::endl;

  std::string f_tagPhase2 = "SiPixelLorentzAngle_phase2_T15_v5_mc";
  std::string l_tagPhase2 = "SiPixelLorentzAngle_phase2_T19_v1_mc";

  SiPixelLorentzAngleValuesBarrelCompareTwoTags histoPhase2;
  histoPhase2.process(connectionString, PI::mk_input(f_tagPhase2, 1, 1, l_tagPhase2, 1, 1));
  edm::LogPrint("testSiPixelPayloadInspector") << histoPhase2.data() << std::endl;

  // 2 tags comparisons

  std::string tag2 = "SiPixelLorentzAngle_2016_ultralegacymc_v2";
  cond::Time_t start2 = boost::lexical_cast<unsigned long long>(1);

  SiPixelLorentzAngleValueComparisonTwoTags histo6;
  histo6.process(connectionString, PI::mk_input(tag, start, start, tag2, start2, start2));
  edm::LogPrint("testSiPixelPayloadInspector") << histo6.data() << std::endl;

  SiPixelLorentzAngleByRegionComparisonTwoTags histo7;
  histo7.process(connectionString, PI::mk_input(tag, start, start, tag2, start2, start2));
  edm::LogPrint("testSiPixelPayloadInspector") << histo7.data() << std::endl;

  // SiPixelQuality

  tag = "SiPixelQuality_forDigitizer_phase1_2018_permanentlyBad";
  start = boost::lexical_cast<unsigned long long>(1);
  end = boost::lexical_cast<unsigned long long>(1);

  edm::LogPrint("testSiPixelPayloadInspector") << "## Exercising SiPixelQuality plots " << std::endl;

  SiPixelBPixQualityMap histo8;
  histo8.process(connectionString, PI::mk_input(tag, start, start));
  edm::LogPrint("testSiPixelPayloadInspector") << histo8.data() << std::endl;

  SiPixelFPixQualityMap histo9;
  histo9.process(connectionString, PI::mk_input(tag, start, start));
  edm::LogPrint("testSiPixelPayloadInspector") << histo9.data() << std::endl;

  // SiPixelGainCalibrationOffline

  tag = "SiPixelGainCalibration_2009runs_express";
  start = boost::lexical_cast<unsigned long long>(312203);
  end = boost::lexical_cast<unsigned long long>(312203);

  edm::LogPrint("testSiPixelPayloadInspector") << "## Exercising SiPixelGainCalibrationOffline plots " << std::endl;

  SiPixelGainCalibrationOfflineGainsValues histo10;
  histo10.process(connectionString, PI::mk_input(tag, start, start));
  edm::LogPrint("testSiPixelPayloadInspector") << histo10.data() << std::endl;

  SiPixelGainCalibrationOfflinePedestalsValues histo11;
  histo11.process(connectionString, PI::mk_input(tag, start, start));
  edm::LogPrint("testSiPixelPayloadInspector") << histo11.data() << std::endl;

  SiPixelGainCalibrationOfflineGainsByPart histo12;
  histo12.process(connectionString, PI::mk_input(tag, start, start));
  edm::LogPrint("testSiPixelPayloadInspector") << histo12.data() << std::endl;

  SiPixelGainCalibrationOfflinePedestalsByPart histo13;
  histo13.process(connectionString, PI::mk_input(tag, start, start));
  edm::LogPrint("testSiPixelPayloadInspector") << histo13.data() << std::endl;

  end = boost::lexical_cast<unsigned long long>(326851);

  SiPixelGainCalibOfflinePedestalComparisonSingleTag histo14;
  histo14.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testSiPixelPayloadInspector") << histo14.data() << std::endl;

  SiPixelGainCalibOfflineGainByRegionComparisonSingleTag histo15;
  histo15.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testSiPixelPayloadInspector") << histo15.data() << std::endl;

  SiPixelGainCalibrationOfflineCorrelations histo16;
  histo16.process(connectionString, PI::mk_input(tag, end, end));
  edm::LogPrint("testSiPixelPayloadInspector") << histo16.data() << std::endl;

  boost::python::dict inputs;
  inputs["SetLog"] = "True";  // sets to true, 1,True,Yes will work

  SiPixelGainCalibrationOfflineGainsValuesBarrel histo17;
  histo17.setInputParamValues(inputs);
  histo17.process(connectionString, PI::mk_input(tag, end, end));
  edm::LogPrint("testSiPixelPayloadInspector") << histo17.data() << std::endl;

  // SiPixelTemplates

  tag = "SiPixelTemplateDBObject38Tv3_express";
  start = boost::lexical_cast<unsigned long long>(326083);
  end = boost::lexical_cast<unsigned long long>(326083);

  edm::LogPrint("testSiPixelPayloadInspector") << "## Exercising SiPixelTemplates plots " << std::endl;

  SiPixelTemplateIDsBPixMap histo18;
  histo18.process(connectionString, PI::mk_input(tag, end, end));
  edm::LogPrint("testSiPixelPayloadInspector") << histo18.data() << std::endl;

  SiPixelTemplateLAFPixMap histo19;
  histo19.process(connectionString, PI::mk_input(tag, end, end));
  edm::LogPrint("testSiPixelPayloadInspector") << histo19.data() << std::endl;

  // SiPixelVCal

  tag = "SiPixelVCal_v1";

  edm::LogPrint("testSiPixelPayloadInspector") << "## Exercising SiPixelVCal plots " << std::endl;

  SiPixelVCalValues histo20;
  histo20.process("frontier://FrontierPrep/CMS_CONDITIONS", PI::mk_input(tag, 1, 1));
  edm::LogPrint("testSiPixelPayloadInspector") << histo20.data() << std::endl;

  SiPixelVCalSlopeValuesBarrel histo21;
  histo21.process("frontier://FrontierPrep/CMS_CONDITIONS", PI::mk_input(tag, 1, 1));
  edm::LogPrint("testSiPixelPayloadInspector") << histo21.data() << std::endl;

  SiPixelVCalOffsetValuesEndcap histo22;
  histo22.process("frontier://FrontierPrep/CMS_CONDITIONS", PI::mk_input(tag, 1, 1));
  edm::LogPrint("testSiPixelPayloadInspector") << histo22.data() << std::endl;

  // SiPixelGenErrors

  tag = "SiPixelGenErrorDBObject_phase1_BoR3_HV350_Tr2000";
  start = boost::lexical_cast<unsigned long long>(1);
  end = boost::lexical_cast<unsigned long long>(1);

  edm::LogPrint("testSiPixelPayloadInspector") << "## Exercising SiPixelGenErrors plots " << std::endl;

  SiPixelGenErrorHeaderTable histo23;
  histo23.process(connectionString, PI::mk_input(tag, end, end));
  edm::LogPrint("testSiPixelPayloadInspector") << histo23.data() << std::endl;

  SiPixelGenErrorIDsBPixMap histo24;
  histo24.process(connectionString, PI::mk_input(tag, end, end));
  edm::LogPrint("testSiPixelPayloadInspector") << histo24.data() << std::endl;

  inputs.clear();
#if PY_MAJOR_VERSION >= 3
  // TODO I don't know why this Py_INCREF is necessary...
  Py_INCREF(inputs.ptr());
#endif

  Py_Finalize();
}
