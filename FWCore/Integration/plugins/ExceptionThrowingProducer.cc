#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Integration/plugins/TestServiceOne.h"
#include "FWCore/Integration/plugins/TestServiceTwo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <atomic>
#include <limits>
#include <memory>

constexpr unsigned int kTestStreams = 4;
constexpr unsigned int kUnset = std::numeric_limits<unsigned int>::max();
constexpr unsigned int kNumberOfTestModules = 2;

namespace edmtest {

  namespace {
    struct Cache {};
  }  // namespace

  class ExceptionThrowingProducer
      : public edm::global::EDProducer<edm::StreamCache<Cache>, edm::RunCache<Cache>, edm::LuminosityBlockCache<Cache>> {
  public:
    explicit ExceptionThrowingProducer(edm::ParameterSet const&);

    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

    std::shared_ptr<Cache> globalBeginRun(edm::Run const&, edm::EventSetup const&) const override;
    void globalEndRun(edm::Run const&, edm::EventSetup const&) const override;
    std::shared_ptr<Cache> globalBeginLuminosityBlock(edm::LuminosityBlock const&,
                                                      edm::EventSetup const&) const override;
    void globalEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) const override;

    std::unique_ptr<Cache> beginStream(edm::StreamID) const override;
    void streamBeginRun(edm::StreamID, edm::Run const&, edm::EventSetup const&) const override;
    void streamBeginLuminosityBlock(edm::StreamID, edm::LuminosityBlock const&, edm::EventSetup const&) const override;
    void streamEndLuminosityBlock(edm::StreamID, edm::LuminosityBlock const&, edm::EventSetup const&) const override;
    void streamEndRun(edm::StreamID, edm::Run const&, edm::EventSetup const&) const override;

    void endJob() override;

    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    edm::EventID eventIDThrowOnEvent_;
    edm::EventID eventIDThrowOnGlobalBeginRun_;
    edm::EventID eventIDThrowOnGlobalBeginLumi_;
    edm::EventID eventIDThrowOnGlobalEndRun_;
    edm::EventID eventIDThrowOnGlobalEndLumi_;
    edm::EventID eventIDThrowOnStreamBeginRun_;
    edm::EventID eventIDThrowOnStreamBeginLumi_;
    edm::EventID eventIDThrowOnStreamEndRun_;
    edm::EventID eventIDThrowOnStreamEndLumi_;

    mutable std::vector<unsigned int> nStreamBeginLumi_;
    mutable std::vector<unsigned int> nStreamEndLumi_;

    unsigned int expectedStreamBeginLumi_;
    unsigned int expectedOffsetNoStreamEndLumi_;
    mutable unsigned int streamWithBeginLumiException_ = kUnset;

    mutable std::atomic<bool> streamBeginLumiExceptionOccurred_ = false;
    mutable std::atomic<bool> streamEndLumiExceptionOccurred_ = false;
  };

  ExceptionThrowingProducer::ExceptionThrowingProducer(edm::ParameterSet const& pset)
      : eventIDThrowOnEvent_(pset.getUntrackedParameter<edm::EventID>("eventIDThrowOnEvent")),
        eventIDThrowOnGlobalBeginRun_(pset.getUntrackedParameter<edm::EventID>("eventIDThrowOnGlobalBeginRun")),
        eventIDThrowOnGlobalBeginLumi_(pset.getUntrackedParameter<edm::EventID>("eventIDThrowOnGlobalBeginLumi")),
        eventIDThrowOnGlobalEndRun_(pset.getUntrackedParameter<edm::EventID>("eventIDThrowOnGlobalEndRun")),
        eventIDThrowOnGlobalEndLumi_(pset.getUntrackedParameter<edm::EventID>("eventIDThrowOnGlobalEndLumi")),
        eventIDThrowOnStreamBeginRun_(pset.getUntrackedParameter<edm::EventID>("eventIDThrowOnStreamBeginRun")),
        eventIDThrowOnStreamBeginLumi_(pset.getUntrackedParameter<edm::EventID>("eventIDThrowOnStreamBeginLumi")),
        eventIDThrowOnStreamEndRun_(pset.getUntrackedParameter<edm::EventID>("eventIDThrowOnStreamEndRun")),
        eventIDThrowOnStreamEndLumi_(pset.getUntrackedParameter<edm::EventID>("eventIDThrowOnStreamEndLumi")),
        nStreamBeginLumi_(kTestStreams, 0),
        nStreamEndLumi_(kTestStreams, 0),
        expectedStreamBeginLumi_(pset.getUntrackedParameter<unsigned int>("expectedStreamBeginLumi")),
        expectedOffsetNoStreamEndLumi_(pset.getUntrackedParameter<unsigned int>("expectedOffsetNoStreamEndLumi")) {}

  void ExceptionThrowingProducer::produce(edm::StreamID, edm::Event& event, edm::EventSetup const&) const {
    if (event.id() == eventIDThrowOnEvent_) {
      throw cms::Exception("IntentionalTestException")
          << "ExceptionThrowingProducer::produce, module configured to throw on: " << eventIDThrowOnEvent_;
    }
  }

  std::shared_ptr<Cache> ExceptionThrowingProducer::globalBeginRun(edm::Run const& run, edm::EventSetup const&) const {
    if (edm::EventID(run.id().run(), edm::invalidLuminosityBlockNumber, edm::invalidEventNumber) ==
        eventIDThrowOnGlobalBeginRun_) {
      throw cms::Exception("IntentionalTestException")
          << "ExceptionThrowingProducer::globalBeginRun, module configured to throw on: "
          << eventIDThrowOnGlobalBeginRun_;
    }
    return std::make_shared<Cache>();
  }

  void ExceptionThrowingProducer::globalEndRun(edm::Run const& run, edm::EventSetup const&) const {
    if (edm::EventID(run.id().run(), edm::invalidLuminosityBlockNumber, edm::invalidEventNumber) ==
        eventIDThrowOnGlobalEndRun_) {
      throw cms::Exception("IntentionalTestException")
          << "ExceptionThrowingProducer::globalEndRun, module configured to throw on: " << eventIDThrowOnGlobalEndRun_;
    }
  }

  std::shared_ptr<Cache> ExceptionThrowingProducer::globalBeginLuminosityBlock(edm::LuminosityBlock const& lumi,
                                                                               edm::EventSetup const&) const {
    if (edm::EventID(lumi.id().run(), lumi.id().luminosityBlock(), edm::invalidEventNumber) ==
        eventIDThrowOnGlobalBeginLumi_) {
      throw cms::Exception("IntentionalTestException")
          << "ExceptionThrowingProducer::globalBeginLuminosityBlock, module configured to throw on: "
          << eventIDThrowOnGlobalBeginLumi_;
    }
    return std::make_shared<Cache>();
  }

  void ExceptionThrowingProducer::globalEndLuminosityBlock(edm::LuminosityBlock const& lumi,
                                                           edm::EventSetup const&) const {
    if (edm::EventID(lumi.id().run(), lumi.id().luminosityBlock(), edm::invalidEventNumber) ==
        eventIDThrowOnGlobalEndLumi_) {
      throw cms::Exception("IntentionalTestException")
          << "ExceptionThrowingProducer::globalEndLuminosityBlock, module configured to throw on: "
          << eventIDThrowOnGlobalEndLumi_;
    }
  }

  std::unique_ptr<Cache> ExceptionThrowingProducer::beginStream(edm::StreamID) const {
    return std::make_unique<Cache>();
  }

  void ExceptionThrowingProducer::streamBeginRun(edm::StreamID iStream,
                                                 edm::Run const& run,
                                                 edm::EventSetup const&) const {
    if (iStream.value() == 0 &&
        edm::EventID(run.id().run(), edm::invalidLuminosityBlockNumber, edm::invalidEventNumber) ==
            eventIDThrowOnStreamBeginRun_) {
      throw cms::Exception("IntentionalTestException")
          << "ExceptionThrowingProducer::streamBeginRun, module configured to throw on: "
          << eventIDThrowOnStreamBeginRun_;
    }
  }

  void ExceptionThrowingProducer::streamBeginLuminosityBlock(edm::StreamID iStream,
                                                             edm::LuminosityBlock const& lumi,
                                                             edm::EventSetup const&) const {
    if (iStream < kTestStreams) {
      ++nStreamBeginLumi_[iStream];
    }

    // Throw if this lumi's ID matches the configured ID (this code is written so
    // that only the first stream to match it will throw).
    bool expected = false;
    if (edm::EventID(lumi.run(), lumi.id().luminosityBlock(), edm::invalidEventNumber) ==
            eventIDThrowOnStreamBeginLumi_ &&
        streamBeginLumiExceptionOccurred_.compare_exchange_strong(expected, true)) {
      // Remember which stream threw
      streamWithBeginLumiException_ = iStream.value();

      throw cms::Exception("IntentionalTestException")
          << "ExceptionThrowingProducer::streamBeginLuminosityBlock, module configured to throw on: "
          << eventIDThrowOnStreamBeginLumi_;
    }
  }

  void ExceptionThrowingProducer::streamEndLuminosityBlock(edm::StreamID iStream,
                                                           edm::LuminosityBlock const& lumi,
                                                           edm::EventSetup const&) const {
    if (iStream < kTestStreams) {
      ++nStreamEndLumi_[iStream];
    }

    bool expected = false;
    if (edm::EventID(lumi.run(), lumi.id().luminosityBlock(), edm::invalidEventNumber) ==
            eventIDThrowOnStreamEndLumi_ &&
        streamEndLumiExceptionOccurred_.compare_exchange_strong(expected, true)) {
      throw cms::Exception("IntentionalTestException")
          << "ExceptionThrowingProducer::streamEndLuminosityBlock, module configured to throw on: "
          << eventIDThrowOnStreamEndLumi_;
    }
  }

  void ExceptionThrowingProducer::streamEndRun(edm::StreamID iStream,
                                               edm::Run const& run,
                                               edm::EventSetup const&) const {
    if (iStream.value() == 0 &&
        edm::EventID(run.id().run(), edm::invalidLuminosityBlockNumber, edm::invalidEventNumber) ==
            eventIDThrowOnStreamEndRun_) {
      throw cms::Exception("IntentionalTestException")
          << "ExceptionThrowingProducer::streamEndRun, module configured to throw on: " << eventIDThrowOnStreamEndRun_;
    }
  }

  void ExceptionThrowingProducer::endJob() {
    bool testsPass = true;
    unsigned int totalStreamBeginLumi = 0;

    unsigned int i = 0;
    for (auto const& nStreamBeginLumi : nStreamBeginLumi_) {
      totalStreamBeginLumi += nStreamBeginLumi;

      // Don't know exact number to expect because streams might skip a lumi so
      // only throw if it is greater than the maximum possible and we only know
      // that for sure if the exception was thrown in stream begin lumi.
      if (nStreamBeginLumi > expectedStreamBeginLumi_ && streamWithBeginLumiException_ != kUnset) {
        edm::LogAbsolute("ExceptionThrowingProducer")
            << "FAILED: More than maximum possible number of streamBeginLumi transitions, stream " << i << " saw "
            << nStreamBeginLumi << " max possible " << expectedStreamBeginLumi_;
        testsPass = false;
      }
      unsigned int expectedStreamEndLumi =
          (streamWithBeginLumiException_ == i) ? nStreamBeginLumi - 1 : nStreamBeginLumi;
      if (nStreamEndLumi_[i] != expectedStreamEndLumi) {
        edm::LogAbsolute("ExceptionThrowingProducer")
            << "FAILED: Unexpected number of streamEndLumi transitions, stream " << i << " saw " << nStreamEndLumi_[i]
            << " expected " << expectedStreamEndLumi;
        testsPass = false;
      }

      ++i;
    }

    edm::Service<edmtest::TestServiceOne> serviceOne;
    if (serviceOne->nPreStreamBeginLumi() != totalStreamBeginLumi ||
        serviceOne->nPostStreamBeginLumi() != totalStreamBeginLumi ||
        serviceOne->nPreStreamEndLumi() != totalStreamBeginLumi ||
        serviceOne->nPostStreamEndLumi() != totalStreamBeginLumi ||
        serviceOne->nPreModuleStreamBeginLumi() != totalStreamBeginLumi * kNumberOfTestModules ||
        serviceOne->nPostModuleStreamBeginLumi() != totalStreamBeginLumi * kNumberOfTestModules ||
        serviceOne->nPreModuleStreamEndLumi() !=
            totalStreamBeginLumi * kNumberOfTestModules - expectedOffsetNoStreamEndLumi_ ||
        serviceOne->nPostModuleStreamEndLumi() !=
            totalStreamBeginLumi * kNumberOfTestModules - expectedOffsetNoStreamEndLumi_) {
      edm::LogAbsolute("ExceptionThrowingProducer")
          << "FAILED: Unexpected number of service transitions in TestServiceOne, stream ";
      testsPass = false;
    }

    edm::Service<edmtest::TestServiceTwo> serviceTwo;
    if (serviceTwo->nPreStreamBeginLumi() != totalStreamBeginLumi ||
        serviceTwo->nPostStreamBeginLumi() != totalStreamBeginLumi ||
        serviceTwo->nPreStreamEndLumi() != totalStreamBeginLumi ||
        serviceTwo->nPostStreamEndLumi() != totalStreamBeginLumi ||
        serviceTwo->nPreModuleStreamBeginLumi() != totalStreamBeginLumi * kNumberOfTestModules ||
        serviceTwo->nPostModuleStreamBeginLumi() != totalStreamBeginLumi * kNumberOfTestModules ||
        serviceTwo->nPreModuleStreamEndLumi() !=
            totalStreamBeginLumi * kNumberOfTestModules - expectedOffsetNoStreamEndLumi_ ||
        serviceTwo->nPostModuleStreamEndLumi() !=
            totalStreamBeginLumi * kNumberOfTestModules - expectedOffsetNoStreamEndLumi_) {
      edm::LogAbsolute("ExceptionThrowingProducer")
          << "FAILED: Unexpected number of service transitions in TestServiceTwo, stream ";
      testsPass = false;
    }

    if (testsPass) {
      edm::LogAbsolute("ExceptionThrowingProducer") << "All tests in ExceptionThrowingProducer PASSED";
    } else {
      edm::LogAbsolute("ExceptionThrowingProducer") << "At least one test in ExceptionThrowingProducer FAILED";
    }
  }

  void ExceptionThrowingProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    edm::EventID invalidEventID;
    desc.addUntracked<edm::EventID>("eventIDThrowOnEvent", invalidEventID);
    desc.addUntracked<edm::EventID>("eventIDThrowOnGlobalBeginRun", invalidEventID);
    desc.addUntracked<edm::EventID>("eventIDThrowOnGlobalBeginLumi", invalidEventID);
    desc.addUntracked<edm::EventID>("eventIDThrowOnGlobalEndRun", invalidEventID);
    desc.addUntracked<edm::EventID>("eventIDThrowOnGlobalEndLumi", invalidEventID);
    desc.addUntracked<edm::EventID>("eventIDThrowOnStreamBeginRun", invalidEventID);
    desc.addUntracked<edm::EventID>("eventIDThrowOnStreamBeginLumi", invalidEventID);
    desc.addUntracked<edm::EventID>("eventIDThrowOnStreamEndRun", invalidEventID);
    desc.addUntracked<edm::EventID>("eventIDThrowOnStreamEndLumi", invalidEventID);

    desc.addUntracked<unsigned int>("expectedStreamBeginLumi", kUnset);
    desc.addUntracked<unsigned int>("expectedOffsetNoStreamEndLumi", 0);
    descriptions.addDefault(desc);
  }

}  // namespace edmtest
using edmtest::ExceptionThrowingProducer;
DEFINE_FWK_MODULE(ExceptionThrowingProducer);
