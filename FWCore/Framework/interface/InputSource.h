#ifndef FWCore_Framework_InputSource_h
#define FWCore_Framework_InputSource_h

/*----------------------------------------------------------------------

InputSource: Abstract interface for all input sources. Input
sources are responsible for creating an EventPrincipal, using data
controlled by the source, and external to the EventPrincipal itself.

The InputSource is also responsible for dealing with the "process
name list" contained within the EventPrincipal. Each InputSource has
to know what "process" (HLT, PROD, USER, USER1, etc.) the program is
part of. The InputSource is repsonsible for pushing this process name
onto the end of the process name list.

For now, we specify this process name to the constructor of the
InputSource. This should be improved.

 Some questions about this remain:

   1. What should happen if we "rerun" a process? i.e., if "USER1" is
   already last in our input file, and we run again a job which claims
   to be "USER1", what should happen? For now, we just quietly add
   this to the history.

   2. Do we need to detect a problem with a history like:
         HLT PROD USER1 PROD
   or is it up to the user not to do something silly? Right now, there
   is no protection against such sillyness.

Some examples of InputSource subclasses may be:

 1) EmptySource: creates EventPrincipals which contain no EDProducts.
 2) PoolSource: creates EventPrincipals which "contain" the data
    read from a EDM/ROOT file. This source should provide for delayed loading
    of data, thus the quotation marks around contain.
 3) DAQSource: creats EventPrincipals which contain raw data, as
    delivered by the L1 trigger and event builder.

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ProcessingController.h"
#include "FWCore/Utilities/interface/LuminosityBlockIndex.h"
#include "FWCore/Utilities/interface/RunIndex.h"
#include "FWCore/Utilities/interface/Signal.h"
#include "FWCore/Utilities/interface/get_underlying_safe.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include <memory>
#include <string>
#include <chrono>
#include <mutex>

namespace edm {
  class ActivityRegistry;
  class BranchIDListHelper;
  class ConfigurationDescriptions;
  class HistoryAppender;
  class ParameterSet;
  class ParameterSetDescription;
  class ProcessContext;
  class ProcessHistoryRegistry;
  class ProductRegistry;
  class StreamContext;
  class ModuleCallingContext;
  class SharedResourcesAcquirer;
  class ThinnedAssociationsHelper;

  class InputSource {
  public:
    enum ItemType { IsInvalid, IsStop, IsFile, IsRun, IsLumi, IsEvent, IsRepeat, IsSynchronize };

    enum ProcessingMode { Runs, RunsAndLumis, RunsLumisAndEvents };

    /// Constructor
    explicit InputSource(ParameterSet const&, InputSourceDescription const&);

    /// Destructor
    virtual ~InputSource() noexcept(false);

    InputSource(InputSource const&) = delete;             // Disallow copying and moving
    InputSource& operator=(InputSource const&) = delete;  // Disallow copying and moving

    static void fillDescriptions(ConfigurationDescriptions& descriptions);
    static const std::string& baseType();
    static void fillDescription(ParameterSetDescription& desc);
    static void prevalidate(ConfigurationDescriptions&);

    /// Advances the source to the next item
    ItemType nextItemType();

    /// Read next event
    void readEvent(EventPrincipal& ep, StreamContext&);

    /// Read a specific event
    bool readEvent(EventPrincipal& ep, EventID const&, StreamContext&);

    /// Read next luminosity block Auxilary
    std::shared_ptr<LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary();

    /// Read next run Auxiliary
    std::shared_ptr<RunAuxiliary> readRunAuxiliary();

    /// Read next run (new run)
    void readRun(RunPrincipal& runPrincipal, HistoryAppender& historyAppender);

    /// Read next run (same as a prior run)
    void readAndMergeRun(RunPrincipal& rp);

    /// Read next luminosity block (new lumi)
    void readLuminosityBlock(LuminosityBlockPrincipal& lumiPrincipal, HistoryAppender& historyAppender);

    /// Read next luminosity block (same as a prior lumi)
    void readAndMergeLumi(LuminosityBlockPrincipal& lbp);

    /// Read next file
    std::unique_ptr<FileBlock> readFile();

    /// close current file
    void closeFile(FileBlock*, bool cleaningUpAfterException);

    /// Skip the number of events specified.
    /// Offset may be negative.
    void skipEvents(int offset);

    bool goToEvent(EventID const& eventID);

    /// Begin again at the first event
    void rewind();

    /// Set the run number
    void setRunNumber(RunNumber_t r) { setRun(r); }

    /// Set the luminosity block ID
    void setLuminosityBlockNumber_t(LuminosityBlockNumber_t lb) { setLumi(lb); }

    /// issue an event report
    void issueReports(EventID const& eventID, StreamID streamID);

    /// Register any produced products
    virtual void registerProducts();

    /// Accessors for product registry
    std::shared_ptr<ProductRegistry const> productRegistry() const { return get_underlying_safe(productRegistry_); }
    std::shared_ptr<ProductRegistry>& productRegistry() { return get_underlying_safe(productRegistry_); }

    /// Accessors for process history registry.
    ProcessHistoryRegistry const& processHistoryRegistry() const { return *processHistoryRegistry_; }
    ProcessHistoryRegistry& processHistoryRegistry() { return *processHistoryRegistry_; }

    /// Accessors for branchIDListHelper
    std::shared_ptr<BranchIDListHelper const> branchIDListHelper() const {
      return get_underlying_safe(branchIDListHelper_);
    }
    std::shared_ptr<BranchIDListHelper>& branchIDListHelper() { return get_underlying_safe(branchIDListHelper_); }

    /// Accessors for thinnedAssociationsHelper
    std::shared_ptr<ThinnedAssociationsHelper const> thinnedAssociationsHelper() const {
      return get_underlying_safe(thinnedAssociationsHelper_);
    }
    std::shared_ptr<ThinnedAssociationsHelper>& thinnedAssociationsHelper() {
      return get_underlying_safe(thinnedAssociationsHelper_);
    }

    /// Reset the remaining number of events/lumis to the maximum number.
    void repeat() {
      remainingEvents_ = maxEvents_;
      remainingLumis_ = maxLumis_;
    }

    /// Returns nullptr if no resource shared between the Source and a DelayedReader
    std::pair<SharedResourcesAcquirer*, std::recursive_mutex*> resourceSharedWithDelayedReader();

    /// Accessor for maximum number of events to be read.
    /// -1 is used for unlimited.
    int maxEvents() const { return maxEvents_; }

    /// Accessor for remaining number of events to be read.
    /// -1 is used for unlimited.
    int remainingEvents() const { return remainingEvents_; }

    /// Accessor for maximum number of lumis to be read.
    /// -1 is used for unlimited.
    int maxLuminosityBlocks() const { return maxLumis_; }

    /// Accessor for remaining number of lumis to be read.
    /// -1 is used for unlimited.
    int remainingLuminosityBlocks() const { return remainingLumis_; }

    /// Accessor for 'module' description.
    ModuleDescription const& moduleDescription() const { return moduleDescription_; }

    /// Accessor for Process Configuration
    ProcessConfiguration const& processConfiguration() const { return moduleDescription().processConfiguration(); }

    /// Accessor for global process identifier
    std::string const& processGUID() const { return processGUID_; }

    /// Called by framework at beginning of job
    void doBeginJob();

    /// Called by framework at end of job
    void doEndJob();

    /// Called by framework at beginning of lumi block
    virtual void doBeginLumi(LuminosityBlockPrincipal& lbp, ProcessContext const*);

    /// Called by framework at beginning of run
    virtual void doBeginRun(RunPrincipal& rp, ProcessContext const*);

    /// Accessor for the current time, as seen by the input source
    Timestamp const& timestamp() const { return time_; }

    /// Accessor for the reduced process history ID of the current run.
    /// This is the ID of the input process history which does not include
    /// the current process.
    ProcessHistoryID const& reducedProcessHistoryID() const;

    /// Accessor for current run number
    RunNumber_t run() const;

    /// Accessor for current luminosity block number
    LuminosityBlockNumber_t luminosityBlock() const;

    /// RunsLumisAndEvents (default), RunsAndLumis, or Runs.
    ProcessingMode processingMode() const { return processingMode_; }

    /// Accessor for Activity Registry
    std::shared_ptr<ActivityRegistry> actReg() const { return actReg_; }

    /// Called by the framework to merge or insert run in principal cache.
    std::shared_ptr<RunAuxiliary> runAuxiliary() const { return runAuxiliary_; }

    /// Called by the framework to merge or insert lumi in principal cache.
    std::shared_ptr<LuminosityBlockAuxiliary> luminosityBlockAuxiliary() const { return lumiAuxiliary_; }

    bool randomAccess() const;
    ProcessingController::ForwardState forwardState() const;
    ProcessingController::ReverseState reverseState() const;

    class EventSourceSentry {
    public:
      EventSourceSentry(InputSource const& source, StreamContext& sc);
      ~EventSourceSentry();

      EventSourceSentry(EventSourceSentry const&) = delete;             // Disallow copying and moving
      EventSourceSentry& operator=(EventSourceSentry const&) = delete;  // Disallow copying and moving

    private:
      InputSource const& source_;
      StreamContext& sc_;
    };

    class LumiSourceSentry {
    public:
      LumiSourceSentry(InputSource const& source, LuminosityBlockIndex id);
      ~LumiSourceSentry();

      LumiSourceSentry(LumiSourceSentry const&) = delete;             // Disallow copying and moving
      LumiSourceSentry& operator=(LumiSourceSentry const&) = delete;  // Disallow copying and moving

    private:
      InputSource const& source_;
      LuminosityBlockIndex index_;
    };

    class RunSourceSentry {
    public:
      RunSourceSentry(InputSource const& source, RunIndex id);
      ~RunSourceSentry();

      RunSourceSentry(RunSourceSentry const&) = delete;             // Disallow copying and moving
      RunSourceSentry& operator=(RunSourceSentry const&) = delete;  // Disallow copying and moving

    private:
      InputSource const& source_;
      RunIndex index_;
    };

    class FileOpenSentry {
    public:
      typedef signalslot::Signal<void(std::string const&, bool)> Sig;
      explicit FileOpenSentry(InputSource const& source, std::string const& lfn, bool usedFallback);
      ~FileOpenSentry();

      FileOpenSentry(FileOpenSentry const&) = delete;             // Disallow copying and moving
      FileOpenSentry& operator=(FileOpenSentry const&) = delete;  // Disallow copying and moving

    private:
      Sig& post_;
      std::string const& lfn_;
      bool usedFallback_;
    };

    class FileCloseSentry {
    public:
      typedef signalslot::Signal<void(std::string const&, bool)> Sig;
      explicit FileCloseSentry(InputSource const& source, std::string const& lfn, bool usedFallback);
      ~FileCloseSentry();

      FileCloseSentry(FileCloseSentry const&) = delete;             // Disallow copying and moving
      FileCloseSentry& operator=(FileCloseSentry const&) = delete;  // Disallow copying and moving

    private:
      Sig& post_;
      std::string const& lfn_;
      bool usedFallback_;
    };

    signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> preEventReadFromSourceSignal_;
    signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> postEventReadFromSourceSignal_;

  protected:
    virtual void skip(int offset);

    /// To set the current time, as seen by the input source
    void setTimestamp(Timestamp const& theTime) { time_ = theTime; }

    ProductRegistry& productRegistryUpdate() { return *productRegistry_; }
    ProcessHistoryRegistry& processHistoryRegistryForUpdate() { return *processHistoryRegistry_; }
    ItemType state() const { return state_; }
    void setRunAuxiliary(RunAuxiliary* rp) {
      runAuxiliary_.reset(rp);
      newRun_ = newLumi_ = true;
    }
    void setLuminosityBlockAuxiliary(LuminosityBlockAuxiliary* lbp) {
      lumiAuxiliary_.reset(lbp);
      newLumi_ = true;
    }
    void resetRunAuxiliary(bool isNewRun = true) const {
      runAuxiliary_.reset();
      newRun_ = newLumi_ = isNewRun;
    }
    void resetLuminosityBlockAuxiliary(bool isNewLumi = true) const {
      lumiAuxiliary_.reset();
      newLumi_ = isNewLumi;
    }
    void reset() const {
      resetLuminosityBlockAuxiliary();
      resetRunAuxiliary();
      state_ = IsInvalid;
    }
    bool newRun() const { return newRun_; }
    void setNewRun() { newRun_ = true; }
    void resetNewRun() { newRun_ = false; }
    bool newLumi() const { return newLumi_; }
    void setNewLumi() { newLumi_ = true; }
    void resetNewLumi() { newLumi_ = false; }
    bool eventCached() const { return eventCached_; }
    /// Called by the framework to merge or ached() const {return eventCached_;}
    void setEventCached() { eventCached_ = true; }
    void resetEventCached() { eventCached_ = false; }

    ///Called by inheriting classes when running multicore when the receiver has told them to
    /// skip some events.
    void decreaseRemainingEventsBy(int iSkipped);

    ///Begin protected makes it easier to do template programming
    virtual void beginJob();

  private:
    bool eventLimitReached() const { return remainingEvents_ == 0; }
    bool lumiLimitReached() const {
      if (remainingLumis_ == 0) {
        return true;
      }
      if (maxSecondsUntilRampdown_ <= 0) {
        return false;
      }
      auto end = std::chrono::steady_clock::now();
      auto elapsed = end - processingStart_;
      if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() > maxSecondsUntilRampdown_) {
        return true;
      }
      return false;
    }
    bool limitReached() const { return eventLimitReached() || lumiLimitReached(); }
    virtual ItemType getNextItemType() = 0;
    ItemType nextItemType_();
    virtual std::shared_ptr<RunAuxiliary> readRunAuxiliary_() = 0;
    virtual std::shared_ptr<LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_() = 0;
    virtual void readRun_(RunPrincipal& runPrincipal);
    virtual void readLuminosityBlock_(LuminosityBlockPrincipal& lumiPrincipal);
    virtual void readEvent_(EventPrincipal& eventPrincipal) = 0;
    virtual bool readIt(EventID const& id, EventPrincipal& eventPrincipal, StreamContext& streamContext);
    virtual std::unique_ptr<FileBlock> readFile_();
    virtual void closeFile_() {}
    virtual bool goToEvent_(EventID const& eventID);
    virtual void setRun(RunNumber_t r);
    virtual void setLumi(LuminosityBlockNumber_t lb);
    virtual void rewind_();
    virtual void endJob();
    virtual std::pair<SharedResourcesAcquirer*, std::recursive_mutex*> resourceSharedWithDelayedReader_();

    virtual bool randomAccess_() const;
    virtual ProcessingController::ForwardState forwardState_() const;
    virtual ProcessingController::ReverseState reverseState_() const;

  private:
    std::shared_ptr<ActivityRegistry> actReg_;  // We do not use propagate_const because the registry itself is mutable.
    int maxEvents_;
    int remainingEvents_;
    int maxLumis_;
    int remainingLumis_;
    int readCount_;
    int maxSecondsUntilRampdown_;
    std::chrono::time_point<std::chrono::steady_clock> processingStart_;
    ProcessingMode processingMode_;
    ModuleDescription const moduleDescription_;
    edm::propagate_const<std::shared_ptr<ProductRegistry>> productRegistry_;
    edm::propagate_const<std::unique_ptr<ProcessHistoryRegistry>> processHistoryRegistry_;
    edm::propagate_const<std::shared_ptr<BranchIDListHelper>> branchIDListHelper_;
    edm::propagate_const<std::shared_ptr<ThinnedAssociationsHelper>> thinnedAssociationsHelper_;
    std::string processGUID_;
    Timestamp time_;
    mutable bool newRun_;
    mutable bool newLumi_;
    bool eventCached_;
    mutable ItemType state_;
    mutable std::shared_ptr<RunAuxiliary> runAuxiliary_;
    mutable std::shared_ptr<LuminosityBlockAuxiliary> lumiAuxiliary_;
    std::string statusFileName_;

    unsigned int numberOfEventsBeforeBigSkip_;
  };
}  // namespace edm

#endif
