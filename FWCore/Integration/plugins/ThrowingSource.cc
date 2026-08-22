#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Sources/interface/IDGeneratorSourceBase.h"
namespace edm {
  class ThrowingSource final : public IDGeneratorSourceBase<InputSource> {
  public:
    explicit ThrowingSource(ParameterSet const&, InputSourceDescription const&);
    ~ThrowingSource() noexcept(false) final;

    void beginJob(ProductRegistry const&) final;
    void endJob() final;
    void readRun_(RunPrincipal& runPrincipal) final;
    void readLuminosityBlock_(LuminosityBlockPrincipal& lumiPrincipal) final;
    std::shared_ptr<edm::FileBlock> readFile_() final;
    void closeFile_() final;
    std::shared_ptr<edm::RunAuxiliary> readRunAuxiliary_() final;
    std::shared_ptr<edm::LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_() final;
    void readEvent_(edm::EventPrincipal&) final;

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    enum {
      kDoNotThrow,
      kConstructor,
      kReadFile,
      kBeginJob,
      kGetNextItemType,
      kReadRunAuxiliary,
      kReadRun,
      kReadLuminosityBlockAuxiliary,
      kReadLumi,
      kReadEvent,
      kCloseFile,
      kEndJob,
      kDestructor
    };
    bool setRunAndEventInfo(EventID& id, TimeValue_t& time, edm::EventAuxiliary::ExperimentType& eType) final;

    // To test exception throws from sources
    int whenToThrow_;
  };

  ThrowingSource::ThrowingSource(ParameterSet const& pset, InputSourceDescription const& desc)
      : IDGeneratorSourceBase<InputSource>(pset, desc, false),
        whenToThrow_(pset.getUntrackedParameter<int>("whenToThrow", kDoNotThrow)) {
    if (whenToThrow_ == kConstructor)
      throw cms::Exception("TestThrow") << "ThrowingSource constructor";
  }

  void ThrowingSource::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    IDGeneratorSourceBase<InputSource>::fillDescription(desc);
    desc.addUntracked<int>("whenToThrow", kDoNotThrow);
    descriptions.addDefault(desc);
  }

  ThrowingSource::~ThrowingSource() noexcept(false) {
    if (whenToThrow_ == kDestructor)
      throw cms::Exception("TestThrow") << "ThrowingSource destructor";
  }

  //Called from IDGeneratorSourceBase::getNextItemType
  bool ThrowingSource::setRunAndEventInfo(EventID&, TimeValue_t&, edm::EventAuxiliary::ExperimentType&) {
    if (whenToThrow_ == kGetNextItemType) {
      throw cms::Exception("TestThrow") << "ThrowingSource::getNextItemType";
    }
    return true;
  }

  void ThrowingSource::beginJob(edm::ProductRegistry const&) {
    if (whenToThrow_ == kBeginJob)
      throw cms::Exception("TestThrow") << "ThrowingSource::beginJob";
  }

  void ThrowingSource::endJob() {
    if (whenToThrow_ == kEndJob)
      throw cms::Exception("TestThrow") << "ThrowingSource::endJob";
  }

  void ThrowingSource::readLuminosityBlock_(LuminosityBlockPrincipal& lb) {
    if (whenToThrow_ == kReadLumi)
      throw cms::Exception("TestThrow") << "ThrowingSource::beginLuminosityBlock";
  }

  void ThrowingSource::readRun_(RunPrincipal& run) {
    if (whenToThrow_ == kReadRun)
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
