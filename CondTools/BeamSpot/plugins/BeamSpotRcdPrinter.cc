#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"
#include "CondFormats/Common/interface/Time.h"
#include "CondFormats/Common/interface/TimeConversions.h"

#include <TROOT.h>
#include <TSystem.h>
#include <TCanvas.h>
#include <TFile.h>
#include <TLegend.h>
#include <TGraph.h>
#include <TH1.h>

namespace BSPrintUtils {
  std::pair<unsigned int, unsigned int> unpack(cond::Time_t since) {
    auto kLowMask = 0XFFFFFFFF;
    auto run = (since >> 32);
    auto lumi = (since & kLowMask);
    return std::make_pair(run, lumi);
  }
}  // namespace BSPrintUtils

class BeamSpotRcdPrinter : public edm::one::EDAnalyzer<> {
public:
  explicit BeamSpotRcdPrinter(const edm::ParameterSet& iConfig);
  ~BeamSpotRcdPrinter() override;
  void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) override;
  void endJob() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  cond::persistency::ConnectionPool m_connectionPool;
  std::string m_condDb;
  std::string m_tagName;

  // Manually specify the start/end time.
  unsigned long long m_startTime;
  unsigned long long m_endTime;
  // Specify output text file name. Leave empty if do not want to dump beamspots in a file.
  std::string m_output;
};

BeamSpotRcdPrinter::BeamSpotRcdPrinter(const edm::ParameterSet& iConfig)
    : m_connectionPool(),
      m_condDb(iConfig.getParameter<std::string>("conditionDatabase")),
      m_tagName(iConfig.getParameter<std::string>("tagName")),
      m_startTime(iConfig.getParameter<unsigned long long>("startIOV")),
      m_endTime(iConfig.getParameter<unsigned long long>("endIOV")),
      m_output(iConfig.getParameter<std::string>("output")) {
  m_connectionPool.setParameters(iConfig.getParameter<edm::ParameterSet>("DBParameters"));
  m_connectionPool.configure();
}

BeamSpotRcdPrinter::~BeamSpotRcdPrinter() {}

void BeamSpotRcdPrinter::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) {
  cond::Time_t startIov = m_startTime;
  cond::Time_t endIov = m_endTime;
  if (startIov > endIov)
    throw cms::Exception("endTime must be greater than startTime!");
  edm::LogInfo("BeamSpotRcdPrinter") << "[BeamSpotRcdPrinter::" << __func__ << "] "
                                     << "Set start time " << startIov << "\n ... Set end time " << endIov;

  // open db session
  edm::LogInfo("BeamSpotRcdPrinter") << "[BeamSpotRcdPrinter::" << __func__ << "] "
                                     << "Query the condition database " << m_condDb;
  cond::persistency::Session condDbSession = m_connectionPool.createSession(m_condDb);
  condDbSession.transaction().start(true);

  std::stringstream ss;
  // list of times with new IOVs within the time range
  std::vector<cond::Time_t> vTime;

  // query the database
  edm::LogInfo("BeamSpotRcdPrinter") << "[BeamSpotRcdPrinter::" << __func__ << "] "
                                     << "Reading IOVs from tag " << m_tagName;
  cond::persistency::IOVProxy iovProxy = condDbSession.readIov(m_tagName);  // load all?
  auto iovs = iovProxy.selectAll();
  auto iiov = iovs.find(startIov);
  auto eiov = iovs.find(endIov);
  int niov = 0;
  while (iiov != iovs.end() && (*iiov).since <= (*eiov).since) {
    // convert cond::Time_t to seconds since epoch
    if ((*iiov).since < startIov) {
      vTime.push_back(startIov);
    } else {
      vTime.push_back((*iiov).since);
    }
    auto payload = condDbSession.fetchPayload<BeamSpotObjects>((*iiov).payloadId);
    auto runLS = BSPrintUtils::unpack((*iiov).since);
    // print IOVs summary
    ss << runLS.first << "," << runLS.second << " (" << (*iiov).since << ")"
       << " [hash: " << (*iiov).payloadId << "] \n"
       << *payload << std::endl;

    ++iiov;
    ++niov;
  }

  vTime.push_back(endIov);  // used to compute last IOV duration

  edm::LogInfo("BeamSpotRcdPrinter") << "[BeamSpotRcdPrinter::" << __func__ << "] "
                                     << "Read " << niov << " IOVs from tag " << m_tagName
                                     << " corresponding to the specified time interval.\n\n"
                                     << ss.str();

  condDbSession.transaction().commit();

  if (!m_output.empty()) {
    std::ofstream fout;
    fout.open(m_output);
    fout << ss.str();
    fout.close();
  }
}

void BeamSpotRcdPrinter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("conditionDatabase", "frontier://FrontierProd/CMS_CONDITIONS");
  desc.add<std::string>("tagName", "BeamSpotObjects_PCL_byLumi_v0_prompt");
  desc.add<unsigned long long>("startIOV", 1406859487478481);
  desc.add<unsigned long long>("endIOV", 1406876667347162);
  desc.add<std::string>("output", "summary.txt");
  desc.add<std::string>("connect", "");

  edm::ParameterSetDescription descDBParameters;
  descDBParameters.addUntracked<std::string>("authenticationPath", "");
  descDBParameters.addUntracked<int>("authenticationSystem", 0);
  descDBParameters.addUntracked<std::string>("security", "");
  descDBParameters.addUntracked<int>("messageLevel", 0);

  desc.add<edm::ParameterSetDescription>("DBParameters", descDBParameters);
  descriptions.add("BeamSpotRcdPrinter", desc);
}

void BeamSpotRcdPrinter::endJob() {}

DEFINE_FWK_MODULE(BeamSpotRcdPrinter);
