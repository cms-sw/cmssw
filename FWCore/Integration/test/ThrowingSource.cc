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
    ~ThrowingSource();

    virtual void beginJob();
    virtual void endJob();
    virtual void beginLuminosityBlock(edm::LuminosityBlock&);
    virtual void endLuminosityBlock(edm::LuminosityBlock&);
    virtual void beginRun(edm::Run&);
    virtual void endRun(edm::Run&);
    virtual std::unique_ptr<edm::FileBlock> readFile_();
    virtual void closeFile_();
    virtual boost::shared_ptr<edm::RunAuxiliary> readRunAuxiliary_();
    virtual boost::shared_ptr<edm::LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_();
    virtual void readEvent_(edm::EventPrincipal&);
  private:
    enum {
      kDoNotThrow  = 0,
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
    virtual bool setRunAndEventInfo(EventID& id, TimeValue_t& time);
    virtual void produce(Event &);

    // To test exception throws from sources
    int whenToThrow_;
  };

  ThrowingSource::ThrowingSource(ParameterSet const& pset,
				       InputSourceDescription const& desc) :
    ProducerSourceBase(pset, desc, false), whenToThrow_(pset.getUntrackedParameter<int>("whenToThrow", kDoNotThrow)) {
    if (whenToThrow_ == kConstructor) throw cms::Exception("TestThrow") << "ThrowingSource constructor";

  }

  ThrowingSource::~ThrowingSource() {
    if (whenToThrow_ == kDestructor) throw cms::Exception("TestThrow") << "ThrowingSource destructor";
  }

  bool
  ThrowingSource::setRunAndEventInfo(EventID&, TimeValue_t&) {
    return true;
  }

  void
  ThrowingSource::produce(edm::Event&) {
  }

  void
  ThrowingSource::beginJob() {
    if (whenToThrow_ == kBeginJob) throw cms::Exception("TestThrow") << "ThrowingSource::beginJob";
  }

  void
  ThrowingSource::endJob() {
    if (whenToThrow_ == kEndJob) throw cms::Exception("TestThrow") << "ThrowingSource::endJob";
  }

  void
  ThrowingSource::beginLuminosityBlock(LuminosityBlock& lb) {
    if (whenToThrow_ == kBeginLumi) throw cms::Exception("TestThrow") << "ThrowingSource::beginLuminosityBlock";
  }

  void
  ThrowingSource::endLuminosityBlock(LuminosityBlock& lb) {
    if (whenToThrow_ == kEndLumi) throw cms::Exception("TestThrow") << "ThrowingSource::endLuminosityBlock";
  }

  void
  ThrowingSource::beginRun(Run& run) {
    if (whenToThrow_ == kBeginRun) throw cms::Exception("TestThrow") << "ThrowingSource::beginRun";
  }

  void
  ThrowingSource::endRun(Run& run) {
    if (whenToThrow_ == kEndRun) throw cms::Exception("TestThrow") << "ThrowingSource::endRun";
  }

  std::unique_ptr<FileBlock>
  ThrowingSource::readFile_() {
    if (whenToThrow_ == kReadFile) throw cms::Exception("TestThrow") << "ThrowingSource::readFile_";
    return std::unique_ptr<FileBlock>(new FileBlock);
  }

  void
  ThrowingSource::closeFile_() {
    if (whenToThrow_ == kCloseFile) throw cms::Exception("TestThrow") << "ThrowingSource::closeFile_";
  }

  boost::shared_ptr<RunAuxiliary>
  ThrowingSource::readRunAuxiliary_() {
    if (whenToThrow_ == kReadRunAuxiliary) throw cms::Exception("TestThrow") << "ThrowingSource::readRunAuxiliary_";
    Timestamp ts = Timestamp(presentTime());
    resetNewRun();
    return boost::shared_ptr<RunAuxiliary>(new RunAuxiliary(eventID().run(), ts, Timestamp::invalidTimestamp()));
  }

  boost::shared_ptr<LuminosityBlockAuxiliary>
  ThrowingSource::readLuminosityBlockAuxiliary_() {
    if (whenToThrow_ == kReadLuminosityBlockAuxiliary) throw cms::Exception("TestThrow") << "ThrowingSource::readLuminosityBlockAuxiliary_";
    if (processingMode() == Runs) return boost::shared_ptr<LuminosityBlockAuxiliary>();
    Timestamp ts = Timestamp(presentTime());
    resetNewLumi();
    return boost::shared_ptr<LuminosityBlockAuxiliary>(new LuminosityBlockAuxiliary(eventID().run(), eventID().luminosityBlock(), ts, Timestamp::invalidTimestamp()));
  }

  void
  ThrowingSource::readEvent_(EventPrincipal& eventPrincipal) {
    if (whenToThrow_ == kReadEvent) throw cms::Exception("TestThrow") << "ThrowingSource::readEvent_";
    assert(eventCached() || processingMode() != RunsLumisAndEvents);
    EventSourceSentry sentry(*this);
    EventAuxiliary aux(eventID(), processGUID(), Timestamp(presentTime()), false, EventAuxiliary::Undefined);
    eventPrincipal.fillEventPrincipal(aux, processHistoryRegistryForUpdate());
  }
}

using edm::ThrowingSource;
DEFINE_FWK_INPUT_SOURCE(ThrowingSource);

