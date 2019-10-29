/*
 * \file QualityTester.cc
 *
 * Helping EDAnalyzer running the quality tests for clients when:
 * - they receive ME data from the SM 
 * - they are run together with the producers (standalone mode)
 *
 * \author M. Zanetti - CERN PH
 *
 */

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include <cmath>
#include <memory>
#include <string>
#include <cstdio>
#include <sstream>
#include <iostream>

class QualityTester : public DQMEDHarvester {
public:
  typedef dqm::harvesting::DQMStore DQMStore;
  typedef dqm::harvesting::MonitorElement MonitorElement;

  /// Constructor
  QualityTester(const edm::ParameterSet& ps);

  /// Destructor
  ~QualityTester() override;

protected:
  // not called for Harvester for now, might enable that later.
  void analyze(const edm::Event& e, const edm::EventSetup& c) /* override */;

  /// perform the actual quality tests
  void dqmEndLuminosityBlock(DQMStore::IBooker&,
                             DQMStore::IGetter&,
                             edm::LuminosityBlock const& lumiSeg,
                             edm::EventSetup const& c) override;
  void endRun(const edm::Run& r, const edm::EventSetup& c) override;
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;

private:
  void performTests();

  int nEvents;
  int prescaleFactor;
  bool getQualityTestsFromFile;
  std::string Label;
  bool testInEventloop;
  bool qtestOnEndRun;
  bool qtestOnEndJob;
  bool qtestOnEndLumi;
  std::string reportThreshold;
  bool verboseQT;

  DQMStore* bei;

  void configureTests(std::string const& file);
  void attachTests();
};

QualityTester::QualityTester(const edm::ParameterSet& ps) {
  prescaleFactor = ps.getUntrackedParameter<int>("prescaleFactor", 1);
  getQualityTestsFromFile = ps.getUntrackedParameter<bool>("getQualityTestsFromFile", true);
  Label = ps.getUntrackedParameter<std::string>("label", "");
  reportThreshold = ps.getUntrackedParameter<std::string>("reportThreshold", "");
  testInEventloop = ps.getUntrackedParameter<bool>("testInEventloop", false);
  qtestOnEndRun = ps.getUntrackedParameter<bool>("qtestOnEndRun", true);
  qtestOnEndJob = ps.getUntrackedParameter<bool>("qtestOnEndJob", false);
  qtestOnEndLumi = ps.getUntrackedParameter<bool>("qtestOnEndLumi", false);
  verboseQT = ps.getUntrackedParameter<bool>("verboseQT", true);

  bei = &*edm::Service<DQMStore>();

  if (getQualityTestsFromFile) {
    edm::FileInPath qtlist = ps.getUntrackedParameter<edm::FileInPath>("qtList");
    configureTests(qtlist.fullPath());
  } else {
    assert(!"Reading from DB no longer supported.");
  }

  nEvents = 0;
}

QualityTester::~QualityTester() {}

void QualityTester::analyze(const edm::Event& e, const edm::EventSetup& c) {
  if (testInEventloop) {
    nEvents++;
    if (prescaleFactor > 0 && nEvents % prescaleFactor == 0) {
      performTests();
    }
  }
}

void QualityTester::dqmEndLuminosityBlock(DQMStore::IBooker&,
                                          DQMStore::IGetter&,
                                          edm::LuminosityBlock const& lumiSeg,
                                          edm::EventSetup const& context) {
  if (!testInEventloop && qtestOnEndLumi) {
    if (prescaleFactor > 0 && lumiSeg.id().luminosityBlock() % prescaleFactor == 0) {
      performTests();
    }
  }
}

void QualityTester::endRun(const edm::Run& r, const edm::EventSetup& context) {
  if (qtestOnEndRun)
    performTests();
}

void QualityTester::dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) {
  if (qtestOnEndJob)
    performTests();
}

void QualityTester::performTests() {
  // done here because new ME can appear while processing data
  attachTests();

  edm::LogVerbatim("QualityTester") << "Running the Quality Test";

  // TODO: runQTests() on each ME

  if (!reportThreshold.empty()) {
    // map {red, orange, black} -> [QReport message, ...]
    std::map<std::string, std::vector<std::string> > theAlarms;
    // populate from MEs hasError, hasWarning, hasOther

    for (auto& theAlarm : theAlarms) {
      const std::string& alarmType = theAlarm.first;
      const std::vector<std::string>& msgs = theAlarm.second;
      if ((reportThreshold == "black") ||
          (reportThreshold == "orange" && (alarmType == "orange" || alarmType == "red")) ||
          (reportThreshold == "red" && alarmType == "red")) {
        std::cout << std::endl;
        std::cout << "Error Type: " << alarmType << std::endl;
        for (auto const& msg : msgs)
          std::cout << msg << std::endl;
      }
    }
    std::cout << std::endl;
  }
}

void QualityTester::configureTests(std::string const& file) {}

void QualityTester::attachTests() {}

DEFINE_FWK_MODULE(QualityTester);
