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

#include <string>

#include "boost/utility.hpp"
#include "sigc++/signal.h"

#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "FWCore/Framework/interface/ProductRegistryHelper.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

namespace edm {
  class ParameterSet;
  class ConfigurationDescriptions;
  class ParameterSetDescription;
  class ActivityRegistry;

  class InputSource : private ProductRegistryHelper, private boost::noncopyable {
  public:
    enum ItemType {
	IsInvalid,
	IsStop,
	IsFile,
	IsRun,
	IsLumi,
	IsEvent,
	IsRepeat
    };

    enum ProcessingMode {
	Runs,
	RunsAndLumis,
	RunsLumisAndEvents
    };

    typedef ProductRegistryHelper::TypeLabelList TypeLabelList;
    /// Constructor
    explicit InputSource(ParameterSet const&, InputSourceDescription const&);

    /// Destructor
    virtual ~InputSource();

    static void fillDescriptions(ConfigurationDescriptions& descriptions);
    static const std::string& baseType();
    static void fillDescription(ParameterSetDescription & desc);

    ItemType nextItemType();

    /// Read next event
    /// Indicate inability to get a new event by returning a null ptr.
    EventPrincipal* readEvent(boost::shared_ptr<LuminosityBlockPrincipal> lbCache);

    /// Read a specific event
    EventPrincipal* readEvent(EventID const&);

    /// Read next luminosity block Auxilary
    boost::shared_ptr<LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary() {
      return readLuminosityBlockAuxiliary_(); 
    }

    /// Read next run Auxiliary
    boost::shared_ptr<RunAuxiliary> readRunAuxiliary() {return readRunAuxiliary_();}

    /// Read next run
    void readAndCacheRun();

    /// Mark run as read
    int markRun();

    /// Read next luminosity block
    void readAndCacheLumi();

    /// Mark lumi as read
    int markLumi();

    /// Read next file
    boost::shared_ptr<FileBlock> readFile();

    /// close current file
    void closeFile(boost::shared_ptr<FileBlock>);

    /// Skip the number of events specified.
    /// Offset may be negative.
    void skipEvents(int offset);

    /// Begin again at the first event
    void rewind() {
      doneReadAhead_ = false;
      state_ = IsInvalid;
      remainingEvents_ = maxEvents_;
      rewind_();
    }

    /// Wake up the input source
    void wakeUp() {wakeUp_();}

    /// Set the run number
    void setRunNumber(RunNumber_t r) {setRun(r);}

    /// Set the luminosity block ID
    void setLuminosityBlockNumber_t(LuminosityBlockNumber_t lb) {setLumi(lb);}

    /// issue an event report
    void issueReports(EventID const& eventID, LuminosityBlockNumber_t const& lumi);

    /// Register any produced products
    void registerProducts();

    /// Accessor for product registry.
    boost::shared_ptr<ProductRegistry const> productRegistry() const {return productRegistry_;}
    
    /// Reset the remaining number of events/lumis to the maximum number.
    void repeat() {
      remainingEvents_ = maxEvents_;
      remainingLumis_ = maxLumis_;
      doneReadAhead_ = false;
    }

    /// Accessor for maximum number of events to be read.
    /// -1 is used for unlimited.
    int maxEvents() const {return maxEvents_;}

    /// Accessor for remaining number of events to be read.
    /// -1 is used for unlimited.
    int remainingEvents() const {return remainingEvents_;}

    /// Accessor for maximum number of lumis to be read.
    /// -1 is used for unlimited.
    int maxLuminosityBlocks() const {return maxLumis_;}

    /// Accessor for remaining number of lumis to be read.
    /// -1 is used for unlimited.
    int remainingLuminosityBlocks() const {return remainingLumis_;}

    /// Accessor for 'module' description.
    ModuleDescription const& moduleDescription() const {return moduleDescription_;}

    /// Accessor for Process Configuration
    ProcessConfiguration const& processConfiguration() const {return moduleDescription().processConfiguration();}

    /// Accessor for primary input source flag
    bool const primary() const {return primary_;}

    /// Accessor for global process identifier
    std::string const& processGUID() const {return processGUID_;}

    /// Called by framework at beginning of job
    void doBeginJob();

    /// Called by framework at end of job
    void doEndJob();

    /// Called by framework at beginning of lumi block
    void doBeginLumi(LuminosityBlockPrincipal& lbp);

    /// Called by framework at end of lumi block
    void doEndLumi(LuminosityBlockPrincipal& lbp);

    /// Called by framework at beginning of run
    void doBeginRun(RunPrincipal& rp);

    /// Called by framework at end of run
    void doEndRun(RunPrincipal& rp);

    /// Called by the framework before forking the process
    void doPreForkReleaseResources();
    void doPostForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren, unsigned int iNumberOfSequentialChildren);
     
    /// Accessor for the current time, as seen by the input source
    Timestamp const& timestamp() const {return time_;}

    /// Accessor for the input process history ID of the current run
    ProcessHistoryID const&  processHistoryID() const;

    /// Accessor for current run number
    RunNumber_t run() const;

    /// Accessor for current luminosity block number
    LuminosityBlockNumber_t luminosityBlock() const;

    /// RunsLumisAndEvents (default), RunsAndLumis, or Runs.
    ProcessingMode processingMode() const {return processingMode_;}

    /// Accessor for Activity Registry
    boost::shared_ptr<ActivityRegistry> actReg() const {return actReg_;}

    /// Called by the framework to merge or insert run in principal cache.
    boost::shared_ptr<RunAuxiliary> runAuxiliary() const {return runAuxiliary_;}

    /// Called by the framework to merge or insert lumi in principal cache.
    boost::shared_ptr<LuminosityBlockAuxiliary> luminosityBlockAuxiliary() const {return lumiAuxiliary_;}

    /// Called by the framework when the run principal cache is cleared
    void respondToClearingRunCache();

    /// Called by the framework when the lumi principal cache is cleared
    void respondToClearingLumiCache();

    using ProductRegistryHelper::produces;
    using ProductRegistryHelper::typeLabelList;

    class SourceSentry : private boost::noncopyable {
    public:
      typedef sigc::signal<void> Sig;
      SourceSentry(Sig& pre, Sig& post);
      ~SourceSentry();
    private:
      Sig& post_;
    };

    class EventSourceSentry {
    public:
      explicit EventSourceSentry(InputSource const& source);
    private:
      SourceSentry sentry_;
    };

    class LumiSourceSentry {
    public:
      explicit LumiSourceSentry(InputSource const& source);
    private:
      SourceSentry sentry_;
    };

    class RunSourceSentry {
    public:
      explicit RunSourceSentry(InputSource const& source);
    private:
      SourceSentry sentry_;
    };

    class FileOpenSentry {
    public:
      explicit FileOpenSentry(InputSource const& source);
    private:
      SourceSentry sentry_;
    };

    class FileCloseSentry {
    public:
      explicit FileCloseSentry(InputSource const& source);
    private:
      SourceSentry sentry_;
    };

  protected:
    /// To set the current time, as seen by the input source
    void setTimestamp(Timestamp const& theTime) {time_ = theTime;}

    ProductRegistry& productRegistryUpdate() const {return const_cast<ProductRegistry&>(*productRegistry_);}
    ItemType state() const{return state_;}
    void setRunAuxiliary(RunAuxiliary *rp) {runAuxiliary_.reset(rp);}
    void setLuminosityBlockAuxiliary(LuminosityBlockAuxiliary* lbp) {lumiAuxiliary_.reset(lbp);}
    void resetRunAuxiliary() const {
      runAuxiliary_.reset();
    }
    void resetLuminosityBlockAuxiliary() const {
      lumiAuxiliary_.reset();
    }
    void reset() const {
      resetLuminosityBlockAuxiliary();
      resetRunAuxiliary();
      doneReadAhead_ = false;
      state_ = IsInvalid;
    }
    EventPrincipal* const eventPrincipalCache();
    PrincipalCache const& principalCache() const {return *principalCache_;}
    PrincipalCache& principalCache() {return *principalCache_;}
    boost::shared_ptr<LuminosityBlockPrincipal> const luminosityBlockPrincipal() const;
    boost::shared_ptr<RunPrincipal> const runPrincipal() const;

    void setRunPrematurelyRead() {runPrematurelyRead_ = true;}
    void setLumiPrematurelyRead() {lumiPrematurelyRead_ = true;}
  private:
    bool eventLimitReached() const {return remainingEvents_ == 0;}
    bool lumiLimitReached() const {return remainingLumis_ == 0;}
    bool limitReached() const {return eventLimitReached() || lumiLimitReached();}
    virtual ItemType getNextItemType() = 0;
    ItemType nextItemType_();
    virtual boost::shared_ptr<RunAuxiliary> readRunAuxiliary_() = 0;
    virtual boost::shared_ptr<LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_() = 0;
    virtual boost::shared_ptr<RunPrincipal> readRun_(boost::shared_ptr<RunPrincipal> rpCache);
    virtual boost::shared_ptr<LuminosityBlockPrincipal> readLuminosityBlock_(
	boost::shared_ptr<LuminosityBlockPrincipal> lbCache);
    virtual EventPrincipal* readEvent_() = 0;
    virtual EventPrincipal* readIt(EventID const&);
    virtual boost::shared_ptr<FileBlock> readFile_();
    virtual void closeFile_() {}
    virtual void skip(int);
    virtual void setRun(RunNumber_t r);
    virtual void setLumi(LuminosityBlockNumber_t lb);
    virtual void rewind_();
    virtual void wakeUp_();
    void preRead();
    void postRead(Event& event);
    virtual void beginLuminosityBlock(LuminosityBlock&);
    virtual void endLuminosityBlock(LuminosityBlock&);
    virtual void beginRun(Run&);
    virtual void endRun(Run&);
    virtual void beginJob();
    virtual void endJob();
    virtual void preForkReleaseResources();
    virtual void postForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren, unsigned int iNumberOfSequentialChildren);

     
  private:

    boost::shared_ptr<ActivityRegistry> actReg_;
    PrincipalCache* principalCache_;
    int maxEvents_;
    int remainingEvents_;
    int maxLumis_;
    int remainingLumis_;
    int readCount_;
    ProcessingMode processingMode_;
    ModuleDescription const moduleDescription_;
    boost::shared_ptr<ProductRegistry const> productRegistry_;
    bool const primary_;
    std::string processGUID_;
    Timestamp time_;
    mutable bool doneReadAhead_;
    mutable ItemType state_;
    mutable boost::shared_ptr<RunAuxiliary> runAuxiliary_;
    mutable boost::shared_ptr<LuminosityBlockAuxiliary>  lumiAuxiliary_;
    bool runPrematurelyRead_;
    bool lumiPrematurelyRead_;
  };
}

#endif
