#include <iostream>
#include <sstream>
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/BeamSpotPlugins/plugins/SimBeamSpot_PayloadInspector.cc"
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
  cond::Time_t start = static_cast<unsigned long long>(1406876667347162);
  cond::Time_t end = static_cast<unsigned long long>(1488257707672138);

  edm::LogPrint("testBeamSpotPayloadInspector") << "## Exercising BeamSpot plots " << std::endl;

  BeamSpotParameters histoParameters;
  histoParameters.process(connectionString, PI::mk_input(tag, start, start));
  edm::LogPrint("testBeamSpotPayloadInspector") << histoParameters.data() << std::endl;

  BeamSpotParametersDiffSingleTag histoParametersDiff;
  histoParametersDiff.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testBeamSpotPayloadInspector") << histoParametersDiff.data() << std::endl;

  std::string tag1 = "BeamSpotObjects_Realistic25ns_900GeV_2021PilotBeams_v2_mc";
  std::string tag2 = "BeamSpotObjects_Realistic25ns_900GeV_2021PilotBeams_v1_mc";
  start = static_cast<unsigned long long>(1);

  BeamSpotParametersDiffTwoTags histoParametersDiffTwoTags;
  histoParametersDiffTwoTags.process(connectionString, PI::mk_input(tag1, start, start, tag2, start, start));
  edm::LogPrint("testBeamSpotPayloadInspector") << histoParametersDiffTwoTags.data() << std::endl;

  edm::LogPrint("testBeamSpotPayloadInspector") << "## Exercising BeamSpotOnline plots " << std::endl;

  // BeamSpotOnline
  tag = "BeamSpotOnlineTestLegacy";
  start = static_cast<unsigned long long>(1443392479297557);
  end = static_cast<unsigned long long>(1470910334763033);

  edm::LogPrint("") << "###########################################################################\n"
                    << "                              DISCLAIMER\n"
                    << " The following unit test is going to print error messages about\n"
                    << " out-of-range indices for the BeamSpotOnlineParameters.\n"
                    << " This normal and expected, since the test payload has been written \n"
                    << " with an obsolete data layout which doesn't contain few additional \n"
                    << " parameters, see https://github.com/cms-sw/cmssw/pull/35338 for details. \n"
                    << " This tests the catching of exceptions coming from reading not existing \n"
                    << " parameters in the Payload inspector.\n";

  BeamSpotOnlineParameters histoOnlineParameters;
  histoOnlineParameters.process(connectionString, PI::mk_input(tag, start, start));
  edm::LogPrint("testBeamSpotPayloadInspector") << histoOnlineParameters.data() << std::endl;

  BeamSpotOnlineParametersDiffSingleTag histoOnlineParametersDiff;
  histoOnlineParametersDiff.process(connectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testBeamSpotPayloadInspector") << histoOnlineParametersDiff.data() << std::endl;

  tag1 = "BeamSpotOnlineObjects_Ideal_Centered_SLHC_v3_mc";
  tag2 = "BeamSpotOnlineObjects_Realistic25ns_13TeVCollisions_RoundOpticsLowSigmaZ_RunBased_v1_mc";
  start = static_cast<unsigned long long>(4294967297); /*Run 1 : LS 1 (packed: 4294967297) this needs to be per LS */

  BeamSpotOnlineParametersDiffTwoTags histoOnlineParametersDiffTwoTags;
  histoOnlineParametersDiffTwoTags.process(connectionString, PI::mk_input(tag1, start, start, tag2, start, start));
  edm::LogPrint("testBeamSpotPayloadInspector") << histoOnlineParametersDiffTwoTags.data() << std::endl;

  // SimBeamSpot
  std::string prepConnectionString("frontier://FrontierPrep/CMS_CONDITIONS");

  tag = "SimBeamSpotObjects_Realistic25ns13p6TeVEOY2022Collision_v0_mc";
  start = static_cast<unsigned long long>(1);
  end = static_cast<unsigned long long>(1);

  edm::LogPrint("testBeamSpotPayloadInspector") << "## Exercising SimBeamSpot plots " << std::endl;

  SimBeamSpotParameters histoSimParameters;
  histoSimParameters.process(prepConnectionString, PI::mk_input(tag, start, start));
  edm::LogPrint("testBeamSpotPayloadInspector") << histoSimParameters.data() << std::endl;

  SimBeamSpotParametersDiffSingleTag histoSimParametersDiff;
  histoSimParametersDiff.process(prepConnectionString, PI::mk_input(tag, start, end));
  edm::LogPrint("testBeamSpotPayloadInspector") << histoSimParametersDiff.data() << std::endl;

  tag1 = "SimBeamSpotObjects_Realistic25ns13p6TeVEOY2022Collision_v0_mc";
  tag2 = "SimBeamSpotObjects_Realistic25ns13p6TeVEarly2023Collision_v0_mc";
  start = static_cast<unsigned long long>(1);

  SimBeamSpotParametersDiffTwoTags histoSimParametersDiffTwoTags;
  histoSimParametersDiffTwoTags.process(prepConnectionString, PI::mk_input(tag1, start, start, tag2, start, start));
  edm::LogPrint("testBeamSpotPayloadInspector") << histoSimParametersDiffTwoTags.data() << std::endl;

  Py_Finalize();
}
