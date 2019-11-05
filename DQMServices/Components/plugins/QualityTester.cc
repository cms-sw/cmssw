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
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "CondFormats/Common/interface/FileBlob.h"
#include "CondFormats/DataRecord/interface/DQMXMLFileRcd.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/ClientConfig/interface/QTestHandle.h"

#include <cmath>
#include <memory>
#include <string>
#include <cstdio>
#include <sstream>
#include <iostream>

#include "boost/scoped_ptr.hpp"

using namespace edm;
using namespace std;

class QualityTester : public edm::EDAnalyzer {
public:
  typedef dqm::harvesting::DQMStore DQMStore;
  typedef dqm::harvesting::MonitorElement MonitorElement;

  /// Constructor
  QualityTester(const edm::ParameterSet& ps);

  /// Destructor
  ~QualityTester() override;

protected:
  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  /// perform the actual quality tests
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c) override;
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void endRun(const edm::Run& r, const edm::EventSetup& c) override;
  void endJob() override;

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

  QTestHandle* qtHandler;
};


QualityTester::QualityTester(const ParameterSet& ps) {
  prescaleFactor = ps.getUntrackedParameter<int>("prescaleFactor", 1);
  getQualityTestsFromFile = ps.getUntrackedParameter<bool>("getQualityTestsFromFile", true);
  Label = ps.getUntrackedParameter<string>("label", "");
  reportThreshold = ps.getUntrackedParameter<string>("reportThreshold", "");
  testInEventloop = ps.getUntrackedParameter<bool>("testInEventloop", false);
  qtestOnEndRun = ps.getUntrackedParameter<bool>("qtestOnEndRun", true);
  qtestOnEndJob = ps.getUntrackedParameter<bool>("qtestOnEndJob", false);
  qtestOnEndLumi = ps.getUntrackedParameter<bool>("qtestOnEndLumi", false);
  verboseQT = ps.getUntrackedParameter<bool>("verboseQT", true);

  bei = &*edm::Service<DQMStore>();

  qtHandler = new QTestHandle;

  // if you use this module, it's non-sense not to provide the QualityTests.xml
  if (getQualityTestsFromFile) {
    edm::FileInPath qtlist = ps.getUntrackedParameter<edm::FileInPath>("qtList");
    qtHandler->configureTests(FileInPath(qtlist).fullPath(), bei);
  }

  nEvents = 0;
}

void QualityTester::beginRun(const edm::Run& run, const edm::EventSetup& iSetup) {
  // if getQualityTestsFromFile is False, it means that the end-user wants them from the Database
  if (!getQualityTestsFromFile) {
    edm::eventsetup::EventSetupRecordKey recordKey(
        edm::eventsetup::EventSetupRecordKey::TypeTag::findType("DQMXMLFileRcd"));
    if (recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
      throw cms::Exception("Record not found") << "Record \"DQMXMLFileRcd"
                                               << "\" does not exist!" << std::endl;
    }
    //     std::cout << "Reading XML from Database" << std::endl ;
    edm::ESHandle<FileBlob> xmlfile;
    iSetup.get<DQMXMLFileRcd>().get(Label, xmlfile);
    std::unique_ptr<std::vector<unsigned char> > vc((*xmlfile).getUncompressedBlob());
    std::string xmlstr = "";
    for (unsigned char& it : *vc) {
      xmlstr += it;
    }

    qtHandler->configureTests(xmlstr, bei, true);
  }
}

QualityTester::~QualityTester() { delete qtHandler; }

void QualityTester::analyze(const edm::Event& e, const edm::EventSetup& c) {
  if (testInEventloop) {
    nEvents++;
    if (getQualityTestsFromFile && prescaleFactor > 0 && nEvents % prescaleFactor == 0) {
      performTests();
    }
  }
}

void QualityTester::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {
  if (!testInEventloop && qtestOnEndLumi) {
    if (getQualityTestsFromFile && prescaleFactor > 0 && lumiSeg.id().luminosityBlock() % prescaleFactor == 0) {
      performTests();
    }
  }
}

void QualityTester::endRun(const Run& r, const EventSetup& context) {
  if (qtestOnEndRun)
    performTests();
}

void QualityTester::endJob() {
  if (qtestOnEndJob)
    performTests();
}

void QualityTester::performTests() {
  // done here because new ME can appear while processing data
  qtHandler->attachTests(bei, verboseQT);

  edm::LogVerbatim("QualityTester") << "Running the Quality Test";

  bei->runQTests();

  if (!reportThreshold.empty()) {
    std::map<std::string, std::vector<std::string> > theAlarms = qtHandler->checkDetailedQTStatus(bei);

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
