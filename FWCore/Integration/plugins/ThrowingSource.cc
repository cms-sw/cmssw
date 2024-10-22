#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Sources/interface/ProducerSourceBase.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace edm {
  class ThrowingSource : public ProducerSourceBase {
  public:
    explicit ThrowingSource(ParameterSet const&, InputSourceDescription const&);
    ~ThrowingSource() noexcept(false) override;

    void beginJob() override;
    void endJob() override;
    void beginLuminosityBlock(edm::LuminosityBlock&) override;
    void beginRun(edm::Run&) override;
    std::shared_ptr<edm::FileBlock> readFile_() override;
    void closeFile_() override;
    std::shared_ptr<edm::RunAuxiliary> readRunAuxiliary_() override;
    std::shared_ptr<edm::LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_() override;
    void readEvent_(edm::EventPrincipal&) override;

  private:
    enum {
      kDoNotThrow = 0,
      kConstructor = 1,
      kBeginJob = 2,
      kBeginRun = 3,
      kBeginLumi = 4,
      kEndLumi = 5,
      kEndRun = 6,
      kEndJob = 7,
      kGetNextItemType = 8,
      kReadEvent = 9,
      kReadLuminosityBlockAuxiliary = 10,
      kReadRunAuxiliary = 11,
      kReadFile = 12,
      kCloseFile = 13,
      kDestructor = 14
    };
    bool setRunAndEventInfo(EventID& id, TimeValue_t& time, edm::EventAuxiliary::ExperimentType& eType) override;
    void produce(Event&) override;

    // To test exception throws from sources
    int whenToThrow_;
  };

  ThrowingSource::ThrowingSource(ParameterSet const& pset, InputSourceDescription const& desc)
      : ProducerSourceBase(pset, desc, false),
        whenToThrow_(pset.getUntrackedParameter<int>("whenToThrow", kDoNotThrow)) {
    if (whenToThrow_ == kConstructor)
      throw cms::Exception("TestThrow") << "ThrowingSource constructor";
  }

  ThrowingSource::~ThrowingSource() noexcept(false) {
    if (whenToThrow_ == kDestructor)
      throw cms::Exception("TestThrow") << "ThrowingSource destructor";
  }

  bool ThrowingSource::setRunAndEventInfo(EventID&, TimeValue_t&, edm::EventAuxiliary::ExperimentType&) { return true; }

  void ThrowingSource::produce(edm::Event&) {}

  void ThrowingSource::beginJob() {
    if (whenToThrow_ == kBeginJob)
      throw cms::Exception("TestThrow") << "ThrowingSource::beginJob";
  }

  void ThrowingSource::endJob() {
    if (whenToThrow_ == kEndJob)
      throw cms::Exception("TestThrow") << "ThrowingSource::endJob";
  }

  void ThrowingSource::beginLuminosityBlock(LuminosityBlock& lb) {
    if (whenToThrow_ == kBeginLumi)
      throw cms::Exception("TestThrow") << "ThrowingSource::beginLuminosityBlock";
  }

  void ThrowingSource::beginRun(Run& run) {
    if (whenToThrow_ == kBeginRun)
      throw cms::Exception("TestThrow") << "ThrowingSource::beginRun";
  }

  std::shared_ptr<FileBlock> ThrowingSource::readFile_() {
    if (whenToThrow_ == kReadFile)
      throw cms::Exception("TestThrow") << "ThrowingSource::readFile_";
    return std::make_shared<FileBlock>();
  }

  void ThrowingSource::closeFile_() {
    if (whenToThrow_ == kCloseFile)
      throw cms::Exception("TestThrow") << "ThrowingSource::closeFile_";
  }

  std::shared_ptr<RunAuxiliary> ThrowingSource::readRunAuxiliary_() {
    if (whenToThrow_ == kReadRunAuxiliary)
      throw cms::Exception("TestThrow") << "ThrowingSource::readRunAuxiliary_";
    Timestamp ts = Timestamp(presentTime());
    resetNewRun();
    return std::make_shared<RunAuxiliary>(eventID().run(), ts, Timestamp::invalidTimestamp());
  }

  std::shared_ptr<LuminosityBlockAuxiliary> ThrowingSource::readLuminosityBlockAuxiliary_() {
    if (whenToThrow_ == kReadLuminosityBlockAuxiliary)
      throw cms::Exception("TestThrow") << "ThrowingSource::readLuminosityBlockAuxiliary_";
    if (processingMode() == Runs)
      return std::shared_ptr<LuminosityBlockAuxiliary>();
    Timestamp ts = Timestamp(presentTime());
    resetNewLumi();
    return std::make_shared<LuminosityBlockAuxiliary>(
        eventID().run(), eventID().luminosityBlock(), ts, Timestamp::invalidTimestamp());
  }

  void ThrowingSource::readEvent_(EventPrincipal& eventPrincipal) {
    if (whenToThrow_ == kReadEvent)
      throw cms::Exception("TestThrow") << "ThrowingSource::readEvent_";
    assert(eventCached() || processingMode() != RunsLumisAndEvents);
    EventAuxiliary aux(eventID(), processGUID(), Timestamp(presentTime()), false, EventAuxiliary::Undefined);
    auto history = processHistoryRegistry().getMapped(aux.processHistoryID());
    eventPrincipal.fillEventPrincipal(aux, history);
  }
}  // namespace edm

using edm::ThrowingSource;
DEFINE_FWK_INPUT_SOURCE(ThrowingSource);
