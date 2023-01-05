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
#include "FWCore/Integration/interface/ESTestRecords.h"
#include "IOVTestInfo.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"

#include "FWCore/Utilities/interface/Exception.h"

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
  };

  ConcurrentIOVAnalyzer::ConcurrentIOVAnalyzer(edm::ParameterSet const& pset)
      : checkExpectedValues_{pset.getUntrackedParameter<bool>("checkExpectedValues")},
        esTokenFromESSource_{esConsumes(pset.getUntrackedParameter<edm::ESInputTag>("fromSource"))},
        esTokenFromESProducer_{esConsumes(edm::ESInputTag("", "fromESProducer"))} {}

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
    descriptions.addDefault(desc);
  }
}  // namespace edmtest
using namespace edmtest;
DEFINE_FWK_MODULE(ConcurrentIOVAnalyzer);
