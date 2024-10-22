// -*- C++ -*-
//
// Package:    FWCore/Integration
// Class:      ConcurrentIOVAnalyzer
//
/**\class edmtest::ConcurrentIOVAnalyzer

 Description: Used in tests of the concurrent IOV features of the
 EventSetup system
*/
// Original Author:  Chris Jones
//         Created:  Fri Jun 24 19:13:25 EDT 2005

#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Integration/interface/ESTestData.h"
#include "FWCore/Integration/interface/ESTestRecords.h"
#include "FWCore/Integration/interface/IOVTestInfo.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <vector>

namespace edmtest {

  class ConcurrentIOVAnalyzer : public edm::global::EDAnalyzer<> {
  public:
    explicit ConcurrentIOVAnalyzer(edm::ParameterSet const&);

    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    bool checkExpectedValues_;
    edm::ESGetToken<IOVTestInfo, ESTestRecordI> const esTokenFromESSource_;
    edm::ESGetToken<IOVTestInfo, ESTestRecordI> const esTokenFromESProducer_;
    edm::ESGetToken<ESTestDataI, ESTestRecordI> esTokenFromAcquireIntESProducer_;
    std::vector<int> expectedESAcquireTestResults_;
    edm::ESGetToken<ESTestDataI, ESTestRecordI> esTokenUniquePtrTestValue_;
    edm::ESGetToken<ESTestDataI, ESTestRecordI> esTokenLambdaUniquePtrTestValue_;
    int expectedUniquePtrTestValue_;
    edm::ESGetToken<ESTestDataI, ESTestRecordI> esTokenOptionalTestValue_;
    edm::ESGetToken<ESTestDataI, ESTestRecordI> esTokenLambdaOptionalTestValue_;
    int expectedOptionalTestValue_;
  };

  ConcurrentIOVAnalyzer::ConcurrentIOVAnalyzer(edm::ParameterSet const& pset)
      : checkExpectedValues_{pset.getUntrackedParameter<bool>("checkExpectedValues")},
        esTokenFromESSource_{esConsumes(pset.getUntrackedParameter<edm::ESInputTag>("fromSource"))},
        esTokenFromESProducer_{esConsumes(edm::ESInputTag("", "fromESProducer"))},
        expectedESAcquireTestResults_{pset.getUntrackedParameter<std::vector<int>>("expectedESAcquireTestResults")},
        expectedUniquePtrTestValue_{pset.getUntrackedParameter<int>("expectedUniquePtrTestValue")},
        expectedOptionalTestValue_{pset.getUntrackedParameter<int>("expectedOptionalTestValue")} {
    if (!expectedESAcquireTestResults_.empty()) {
      esTokenFromAcquireIntESProducer_ = esConsumes(edm::ESInputTag("", "fromAcquireIntESProducer"));
    }
    if (expectedUniquePtrTestValue_ != 0) {
      esTokenUniquePtrTestValue_ = esConsumes(edm::ESInputTag("", "uniquePtr"));
      esTokenLambdaUniquePtrTestValue_ = esConsumes(edm::ESInputTag("", "uniquePtrLambda"));
    }
    if (expectedOptionalTestValue_ != 0) {
      esTokenOptionalTestValue_ = esConsumes(edm::ESInputTag("", "optional"));
      esTokenLambdaOptionalTestValue_ = esConsumes(edm::ESInputTag("", "optionalLambda"));
    }
  }

  void ConcurrentIOVAnalyzer::analyze(edm::StreamID, edm::Event const& event, edm::EventSetup const& eventSetup) const {
    auto lumiNumber = event.eventAuxiliary().luminosityBlock();

    edm::ESHandle<IOVTestInfo> iovTestInfoFromESSource = eventSetup.getHandle(esTokenFromESSource_);
    edm::ESHandle<IOVTestInfo> iovTestInfoFromESProducer = eventSetup.getHandle(esTokenFromESProducer_);

    ESTestRecordI esTestRecordI = eventSetup.get<ESTestRecordI>();
    edm::ValidityInterval iov = esTestRecordI.validityInterval();

    if (iovTestInfoFromESSource->iovStartLumi_ != iov.first().luminosityBlockNumber() ||
        iovTestInfoFromESSource->iovEndLumi_ != iov.last().luminosityBlockNumber() ||
        iovTestInfoFromESSource->iovIndex_ != esTestRecordI.iovIndex() ||
        iovTestInfoFromESSource->cacheIdentifier_ != esTestRecordI.cacheIdentifier()) {
      throw cms::Exception("TestFailure") << "ConcurrentIOVAnalyzer::analyze,"
                                          << " values read from ESSource do not agree with record";
    }

    if (iovTestInfoFromESProducer->iovStartLumi_ != iov.first().luminosityBlockNumber() ||
        iovTestInfoFromESProducer->iovEndLumi_ != iov.last().luminosityBlockNumber() ||
        iovTestInfoFromESProducer->iovIndex_ != esTestRecordI.iovIndex() ||
        iovTestInfoFromESProducer->cacheIdentifier_ != esTestRecordI.cacheIdentifier()) {
      throw cms::Exception("TestFailure") << "ConcurrentIOVAnalyzer::analyze,"
                                          << " values read from ESProducer do not agree with record";
    }

    // First cacheIdentifier in the test is actually 3 (0, 1, and 2 just are ignored)
    unsigned int cacheIdentifier = esTestRecordI.cacheIdentifier();
    if (cacheIdentifier < expectedESAcquireTestResults_.size()) {
      int testResult = eventSetup.getData(esTokenFromAcquireIntESProducer_).value();
      if (testResult != expectedESAcquireTestResults_[cacheIdentifier]) {
        throw cms::Exception("TestFailure")
            << "ConcurrentIOVAnalyzer::analyze,"
            << " unexpected value for EventSetup acquire test.\n"
            << "Expected = " << expectedESAcquireTestResults_[cacheIdentifier] << " result = " << testResult
            << " cacheIdentifier = " << cacheIdentifier << "\n";
      }
    }

    if (expectedUniquePtrTestValue_ != 0) {
      if (eventSetup.getData(esTokenUniquePtrTestValue_).value() != expectedUniquePtrTestValue_) {
        throw cms::Exception("TestFailure")
            << "ConcurrentIOVAnalyzer::analyze,"
            << " value for unique_ptr test from EventSetup does not match expected value";
      }
      if (eventSetup.getData(esTokenLambdaUniquePtrTestValue_).value() != expectedUniquePtrTestValue_) {
        throw cms::Exception("TestFailure")
            << "ConcurrentIOVAnalyzer::analyze,"
            << " value for lambda unique_ptr test from EventSetup does not match expected value";
      }
    }

    if (expectedOptionalTestValue_ != 0) {
      if (eventSetup.getData(esTokenOptionalTestValue_).value() != expectedOptionalTestValue_) {
        throw cms::Exception("TestFailure") << "ConcurrentIOVAnalyzer::analyze,"
                                            << " value for optional test from EventSetup does not match expected value";
      }
      if (eventSetup.getData(esTokenLambdaOptionalTestValue_).value() != expectedOptionalTestValue_) {
        throw cms::Exception("TestFailure")
            << "ConcurrentIOVAnalyzer::analyze,"
            << " value for lambda optional test from EventSetup does not match expected value";
      }
    }

    if (!checkExpectedValues_) {
      return;
    }

    // cacheIdentifier starts at 2 for beginRun and 3 is next here for the first lumi
    if (lumiNumber == 1) {
      if (iovTestInfoFromESProducer->iovStartLumi_ != 1 || iovTestInfoFromESProducer->iovEndLumi_ != 3 ||
          iovTestInfoFromESProducer->cacheIdentifier_ != 3) {
        throw cms::Exception("TestFailure") << "ConcurrentIOVAnalyzer::analyze,"
                                            << " values read from ESProducer do not agree with expected values";
      }
    }
    if (lumiNumber == 2) {
      if (iovTestInfoFromESProducer->iovStartLumi_ != 1 || iovTestInfoFromESProducer->iovEndLumi_ != 3 ||
          iovTestInfoFromESProducer->cacheIdentifier_ != 3) {
        throw cms::Exception("TestFailure") << "ConcurrentIOVAnalyzer::analyze,"
                                            << " values read from ESProducer do not agree with expected values";
      }
    }
    if (lumiNumber == 3) {
      if (iovTestInfoFromESProducer->iovStartLumi_ != 1 || iovTestInfoFromESProducer->iovEndLumi_ != 3 ||
          iovTestInfoFromESProducer->cacheIdentifier_ != 3) {
        throw cms::Exception("TestFailure") << "ConcurrentIOVAnalyzer::analyze,"
                                            << " values read from ESProducer do not agree with expected values";
      }
    }
    if (lumiNumber == 4) {
      if (iovTestInfoFromESProducer->iovStartLumi_ != 4 || iovTestInfoFromESProducer->iovEndLumi_ != 5 ||
          iovTestInfoFromESProducer->cacheIdentifier_ != 4) {
        throw cms::Exception("TestFailure") << "ConcurrentIOVAnalyzer::analyze,"
                                            << " values read from ESProducer do not agree with expected values";
      }
    }
    if (lumiNumber == 5) {
      if (iovTestInfoFromESProducer->iovStartLumi_ != 4 || iovTestInfoFromESProducer->iovEndLumi_ != 5 ||
          iovTestInfoFromESProducer->cacheIdentifier_ != 4) {
        throw cms::Exception("TestFailure") << "ConcurrentIOVAnalyzer::analyze,"
                                            << " values read from ESProducer do not agree with expected values";
      }
    }
    if (lumiNumber == 6) {
      if (iovTestInfoFromESProducer->iovStartLumi_ != 6 || iovTestInfoFromESProducer->iovEndLumi_ != 6 ||
          iovTestInfoFromESProducer->cacheIdentifier_ != 5) {
        throw cms::Exception("TestFailure") << "ConcurrentIOVAnalyzer::analyze,"
                                            << " values read from ESProducer do not agree with expected values";
      }
    }
    if (lumiNumber == 7) {
      if (iovTestInfoFromESProducer->iovStartLumi_ != 7 || iovTestInfoFromESProducer->iovEndLumi_ != 7 ||
          iovTestInfoFromESProducer->cacheIdentifier_ != 6) {
        throw cms::Exception("TestFailure") << "ConcurrentIOVAnalyzer::analyze,"
                                            << " values read from ESProducer do not agree with expected values";
      }
    }
    if (lumiNumber == 8) {
      if (iovTestInfoFromESProducer->iovStartLumi_ != 8 || iovTestInfoFromESProducer->iovEndLumi_ != 8 ||
          iovTestInfoFromESProducer->cacheIdentifier_ != 7) {
        throw cms::Exception("TestFailure") << "ConcurrentIOVAnalyzer::analyze,"
                                            << " values read from ESProducer do not agree with expected values";
      }
    }
  }

  void ConcurrentIOVAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.addUntracked<bool>("checkExpectedValues", true);
    desc.addUntracked<edm::ESInputTag>("fromSource", edm::ESInputTag("", ""));
    desc.addUntracked<std::vector<int>>("expectedESAcquireTestResults", std::vector<int>());
    desc.addUntracked<int>("expectedUniquePtrTestValue", 0);
    desc.addUntracked<int>("expectedOptionalTestValue", 0);
    descriptions.addDefault(desc);
  }
}  // namespace edmtest
using namespace edmtest;
DEFINE_FWK_MODULE(ConcurrentIOVAnalyzer);
