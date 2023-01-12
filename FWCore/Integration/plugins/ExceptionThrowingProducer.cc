#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <memory>

namespace edmtest {

  namespace {
    struct Cache {};
  }  // namespace

  class ExceptionThrowingProducer
      : public edm::global::EDProducer<edm::StreamCache<Cache>, edm::RunCache<Cache>, edm::LuminosityBlockCache<Cache>> {
  public:
    explicit ExceptionThrowingProducer(edm::ParameterSet const&);

    ~ExceptionThrowingProducer() override;

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
        eventIDThrowOnStreamEndLumi_(pset.getUntrackedParameter<edm::EventID>("eventIDThrowOnStreamEndLumi")) {}

  ExceptionThrowingProducer::~ExceptionThrowingProducer() {}

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
    if (iStream.value() == 0 && edm::EventID(lumi.run(), lumi.id().luminosityBlock(), edm::invalidEventNumber) ==
                                    eventIDThrowOnStreamBeginLumi_) {
      throw cms::Exception("IntentionalTestException")
          << "ExceptionThrowingProducer::streamBeginLuminosityBlock, module configured to throw on: "
          << eventIDThrowOnStreamBeginLumi_;
    }
  }

  void ExceptionThrowingProducer::streamEndLuminosityBlock(edm::StreamID iStream,
                                                           edm::LuminosityBlock const& lumi,
                                                           edm::EventSetup const&) const {
    if (iStream.value() == 0 && edm::EventID(lumi.run(), lumi.id().luminosityBlock(), edm::invalidEventNumber) ==
                                    eventIDThrowOnStreamEndLumi_) {
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
    descriptions.addDefault(desc);
  }

}  // namespace edmtest
using edmtest::ExceptionThrowingProducer;
DEFINE_FWK_MODULE(ExceptionThrowingProducer);
