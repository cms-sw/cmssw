// -*- C++ -*-
//
// Package:    FWCore/Integration
// Class:      RunLumiESAnalyzer
//
/**\class edmtest::RunLumiESAnalyzer

 Description: Used in tests of the EventSetup system,
 particularly testing its support of Run and Lumi
 transitions.
*/
// Original Author:  W. David Dagenhart
//         Created:  18 April 2019

#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Integration/interface/ESTestRecords.h"
#include "IOVTestInfo.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <memory>

namespace {
  struct Cache {
    Cache() : value(0) {}
    //Using mutable since we want to update the value.
    mutable std::atomic<unsigned int> value;
  };

  struct UnsafeCache {
    UnsafeCache() : value(0) {}
    unsigned int value;
  };
}  //end anonymous namespace

namespace edmtest {

  class RunLumiESAnalyzer : public edm::global::EDAnalyzer<edm::StreamCache<UnsafeCache>,
                                                           edm::RunCache<Cache>,
                                                           edm::LuminosityBlockCache<Cache>> {
  public:
    explicit RunLumiESAnalyzer(edm::ParameterSet const&);

    std::unique_ptr<UnsafeCache> beginStream(edm::StreamID) const override;
    void streamBeginRun(edm::StreamID, edm::Run const&, edm::EventSetup const&) const override;
    void streamBeginLuminosityBlock(edm::StreamID, edm::LuminosityBlock const&, edm::EventSetup const&) const override;
    void streamEndLuminosityBlock(edm::StreamID, edm::LuminosityBlock const&, edm::EventSetup const&) const override;
    void streamEndRun(edm::StreamID, edm::Run const&, edm::EventSetup const&) const override;

    std::shared_ptr<Cache> globalBeginRun(edm::Run const&, edm::EventSetup const&) const override;
    void globalEndRun(edm::Run const& iRun, edm::EventSetup const&) const override;
    std::shared_ptr<Cache> globalBeginLuminosityBlock(edm::LuminosityBlock const&,
                                                      edm::EventSetup const&) const override;
    void globalEndLuminosityBlock(edm::LuminosityBlock const& iLB, edm::EventSetup const&) const override;

    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    void checkIOVInfo(edm::EventSetup const& eventSetup,
                      unsigned int run,
                      unsigned int lumiNumber,
                      edm::ESHandle<IOVTestInfo> const& iovTestInfo,
                      const char* functionName) const;

    edm::ESGetToken<IOVTestInfo, ESTestRecordC> const esToken_;
    edm::ESGetToken<IOVTestInfo, ESTestRecordC> const tokenBeginRun_;
    edm::ESGetToken<IOVTestInfo, ESTestRecordC> const tokenBeginLumi_;
    edm::ESGetToken<IOVTestInfo, ESTestRecordC> const tokenEndLumi_;
    edm::ESGetToken<IOVTestInfo, ESTestRecordC> const tokenEndRun_;
  };

  RunLumiESAnalyzer::RunLumiESAnalyzer(edm::ParameterSet const&)
      : esToken_{esConsumes(edm::ESInputTag("", ""))},
        tokenBeginRun_{esConsumes<edm::Transition::BeginRun>(edm::ESInputTag("", ""))},
        tokenBeginLumi_{esConsumes<edm::Transition::BeginLuminosityBlock>(edm::ESInputTag("", ""))},
        tokenEndLumi_{esConsumes<edm::Transition::EndLuminosityBlock>(edm::ESInputTag("", ""))},
        tokenEndRun_{esConsumes<edm::Transition::EndRun>(edm::ESInputTag("", ""))} {}

  std::unique_ptr<UnsafeCache> RunLumiESAnalyzer::beginStream(edm::StreamID iID) const {
    return std::make_unique<UnsafeCache>();
  }

  void RunLumiESAnalyzer::checkIOVInfo(edm::EventSetup const& eventSetup,
                                       unsigned int run,
                                       unsigned int lumiNumber,
                                       edm::ESHandle<IOVTestInfo> const& iovTestInfo,
                                       const char* functionName) const {
    ESTestRecordC recordC = eventSetup.get<ESTestRecordC>();
    edm::ValidityInterval iov = recordC.validityInterval();

    if (iovTestInfo->iovStartRun_ != run || iovTestInfo->iovEndRun_ != run ||
        iovTestInfo->iovStartLumi_ != lumiNumber || iovTestInfo->iovEndLumi_ != lumiNumber) {
      throw cms::Exception("TestFailure")
          << functionName << ": values read from EventSetup do not agree with auxiliary";
    }

    if (iov.first().eventID().run() != run || iov.last().eventID().run() != run ||
        iov.first().luminosityBlockNumber() != lumiNumber || iov.last().luminosityBlockNumber() != lumiNumber) {
      throw cms::Exception("TestFailure") << functionName << ": values from EventSetup IOV do not agree with auxiliary";
    }
  }

  void RunLumiESAnalyzer::streamBeginRun(edm::StreamID, edm::Run const& iRun, edm::EventSetup const& eventSetup) const {
    auto run = iRun.runAuxiliary().run();
    unsigned int lumiNumber = 0;
    edm::ESHandle<IOVTestInfo> iovTestInfo = eventSetup.getHandle(tokenBeginRun_);
    checkIOVInfo(eventSetup, run, lumiNumber, iovTestInfo, "RunLumiESAnalyzer::streamBeginRun");
  }

  void RunLumiESAnalyzer::streamBeginLuminosityBlock(edm::StreamID,
                                                     edm::LuminosityBlock const& iLumi,
                                                     edm::EventSetup const& eventSetup) const {
    auto run = iLumi.luminosityBlockAuxiliary().run();
    unsigned int lumiNumber = iLumi.luminosityBlockAuxiliary().luminosityBlock();
    edm::ESHandle<IOVTestInfo> iovTestInfo = eventSetup.getHandle(tokenBeginLumi_);
    checkIOVInfo(eventSetup, run, lumiNumber, iovTestInfo, "RunLumiESAnalyzer::streamBeginLuminosityBlock");
  }

  void RunLumiESAnalyzer::streamEndLuminosityBlock(edm::StreamID,
                                                   edm::LuminosityBlock const& iLumi,
                                                   edm::EventSetup const& eventSetup) const {
    auto run = iLumi.luminosityBlockAuxiliary().run();
    unsigned int lumiNumber = iLumi.luminosityBlockAuxiliary().luminosityBlock();
    edm::ESHandle<IOVTestInfo> iovTestInfo = eventSetup.getHandle(tokenEndLumi_);
    checkIOVInfo(eventSetup, run, lumiNumber, iovTestInfo, "RunLumiESAnalyzer::streamEndLuminosityBlock");
  }

  void RunLumiESAnalyzer::streamEndRun(edm::StreamID, edm::Run const& iRun, edm::EventSetup const& eventSetup) const {
    auto run = iRun.runAuxiliary().run();
    unsigned int lumiNumber = 4294967295;
    edm::ESHandle<IOVTestInfo> iovTestInfo = eventSetup.getHandle(tokenEndRun_);
    checkIOVInfo(eventSetup, run, lumiNumber, iovTestInfo, "RunLumiESAnalyzer::streamEndRun");
  }

  std::shared_ptr<Cache> RunLumiESAnalyzer::globalBeginRun(edm::Run const& iRun,
                                                           edm::EventSetup const& eventSetup) const {
    auto run = iRun.runAuxiliary().run();
    unsigned int lumiNumber = 0;
    edm::ESHandle<IOVTestInfo> iovTestInfo = eventSetup.getHandle(tokenBeginRun_);
    checkIOVInfo(eventSetup, run, lumiNumber, iovTestInfo, "RunLumiESAnalyzer::globalBeginRun");
    return std::make_shared<Cache>();
  }

  void RunLumiESAnalyzer::globalEndRun(edm::Run const& iRun, edm::EventSetup const& eventSetup) const {
    auto run = iRun.runAuxiliary().run();
    unsigned int lumiNumber = 4294967295;
    edm::ESHandle<IOVTestInfo> iovTestInfo = eventSetup.getHandle(tokenEndRun_);
    checkIOVInfo(eventSetup, run, lumiNumber, iovTestInfo, "RunLumiESAnalyzer::globalEndRun");
  }

  std::shared_ptr<Cache> RunLumiESAnalyzer::globalBeginLuminosityBlock(edm::LuminosityBlock const& iLumi,
                                                                       edm::EventSetup const& eventSetup) const {
    auto run = iLumi.luminosityBlockAuxiliary().run();
    unsigned int lumiNumber = iLumi.luminosityBlockAuxiliary().luminosityBlock();
    edm::ESHandle<IOVTestInfo> iovTestInfo = eventSetup.getHandle(tokenBeginLumi_);
    checkIOVInfo(eventSetup, run, lumiNumber, iovTestInfo, "RunLumiESAnalyzer::globalBeginLuminosityBlock");
    return std::make_shared<Cache>();
  }

  void RunLumiESAnalyzer::globalEndLuminosityBlock(edm::LuminosityBlock const& iLumi,
                                                   edm::EventSetup const& eventSetup) const {
    auto run = iLumi.luminosityBlockAuxiliary().run();
    unsigned int lumiNumber = iLumi.luminosityBlockAuxiliary().luminosityBlock();
    edm::ESHandle<IOVTestInfo> iovTestInfo = eventSetup.getHandle(tokenEndLumi_);
    checkIOVInfo(eventSetup, run, lumiNumber, iovTestInfo, "RunLumiESAnalyzer::globalEndLuminosityBlock");
  }

  void RunLumiESAnalyzer::analyze(edm::StreamID, edm::Event const& event, edm::EventSetup const& eventSetup) const {
    auto run = event.eventAuxiliary().run();
    auto lumiNumber = event.eventAuxiliary().luminosityBlock();
    edm::ESHandle<IOVTestInfo> iovTestInfo = eventSetup.getHandle(esToken_);
    checkIOVInfo(eventSetup, run, lumiNumber, iovTestInfo, "RunLumiESAnalyzer::analyzer");
  }

  void RunLumiESAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    descriptions.addDefault(desc);
  }
}  // namespace edmtest
using namespace edmtest;
DEFINE_FWK_MODULE(RunLumiESAnalyzer);
