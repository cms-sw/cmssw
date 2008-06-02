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
    read from a POOL file. This source should provide for delayed loading
    of data, thus the quotation marks around contain.
 3) DAQInputSource: creats EventPrincipals which contain raw data, as
    delivered by the L1 trigger and event builder. 

$Id: InputSource.h,v 1.30 2007/06/25 23:22:12 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>
#include <string>

#include "boost/shared_ptr.hpp"

#include "FWCore/Utilities/interface/GCCPrerequisite.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "FWCore/Framework/interface/ProductRegistryHelper.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

namespace edm {
  class ParameterSet;

#if GCC_PREREQUISITE(3,4,4)
  class InputSource : private ProductRegistryHelper {
#else
  // Bug in gcc3.2.3 compiler forces public inheritance
  class InputSource : public ProductRegistryHelper {
#endif
  public:
    typedef ProductRegistryHelper::TypeLabelList TypeLabelList;
    /// Constructor
    explicit InputSource(ParameterSet const&, InputSourceDescription const&);

    /// Destructor
    virtual ~InputSource();

    /// Indicate inability to get a new event by returning a null auto_ptr.

    /// Read next event
    std::auto_ptr<EventPrincipal> readEvent(boost::shared_ptr<LuminosityBlockPrincipal> lbp);

    /// Read a specific event
    std::auto_ptr<EventPrincipal> readEvent(EventID const&);

    /// Read next luminosity block
    boost::shared_ptr<LuminosityBlockPrincipal> readLuminosityBlock(boost::shared_ptr<RunPrincipal> rp);

    /// Read next run
    boost::shared_ptr<RunPrincipal> readRun();

    /// Skip the number of events specified.
    /// Offset may be negative.
    void skipEvents(int offset);

    /// Begin again at the first event
    void rewind() {rewind_();}

    /// Wake up the input source
    void wakeUp() {wakeUp_();}

    /// Set the run number
    void setRunNumber(RunNumber_t r) {setRun(r);}

    /// Set the luminosity block ID
    void setLuminosityBlockNumber_t(LuminosityBlockNumber_t lb) {setLumi(lb);}

    /// issue an event report
    void issueReports(EventID const&);

    /// Register any produced products
    void registerProducts();

    /// Accessor for product registry.
    boost::shared_ptr<ProductRegistry const> productRegistry() const {return productRegistry_;}
    
    /// Reset the remaining number of events to the maximum number.
    void repeat() {remainingEvents_ = maxEvents_;}

    /// Reset the remaining number of events to zero.
    void noMoreInput() {remainingEvents_ = 0;}

    /// Accessor for maximum number of events to be read.
    /// -1 is used for unlimited.
    int maxEvents() const {return maxEvents_;}

    /// Accessor for remaining number of events to be read.
    int remainingEvents() const {return remainingEvents_;}

    /// Accessor for 'module' description.
    ModuleDescription const& moduleDescription() const {return moduleDescription_;}

    /// Accessor for Process Configuration
    ProcessConfiguration const& processConfiguration() const {return moduleDescription().processConfiguration();}

    /// Accessor for primary input source flag
    bool const primary() const {return primary_;}

    /// Called by framework at beginning of job
    void doBeginJob(EventSetup const&);

    /// Called by framework at end of job
    void doEndJob();

    /// Called by framework when events are exhausted.
    void doFinishLumi(LuminosityBlockPrincipal& lbp);
    void doFinishRun(RunPrincipal& rp);

    /// Accessor for the current time, as seen by the input source
    Timestamp const& timestamp() const {return time_;}

    using ProductRegistryHelper::produces;
    using ProductRegistryHelper::typeLabelList;

  protected:
    /// To set the current time, as seen by the input source
    void setTimestamp(Timestamp const& theTime) {time_ = theTime;}

    ProductRegistry & productRegistryUpdate() const {return const_cast<ProductRegistry &>(*productRegistry_);}

  private:

    // Indicate inability to get a new event by returning a null
    // auto_ptr.
    virtual std::auto_ptr<EventPrincipal> readEvent_(boost::shared_ptr<LuminosityBlockPrincipal>) = 0;
    virtual std::auto_ptr<EventPrincipal> readIt(EventID const&);
    virtual boost::shared_ptr<RunPrincipal> readRun_() = 0;
    virtual boost::shared_ptr<LuminosityBlockPrincipal> readLuminosityBlock_(boost::shared_ptr<RunPrincipal>) = 0;
    virtual void skip(int);
    virtual void setRun(RunNumber_t r);
    virtual void setLumi(LuminosityBlockNumber_t lb);
    virtual void rewind_();
    virtual void wakeUp_();
    void preRead();
    void postRead(Event& event);
    virtual void endLuminosityBlock(LuminosityBlock &);
    virtual void endRun(Run &);
    virtual void beginJob(EventSetup const&);
    virtual void endJob();

  private:

    int maxEvents_;
    int remainingEvents_;
    int readCount_;
    bool unlimited_;
    ModuleDescription const moduleDescription_;
    boost::shared_ptr<ProductRegistry const> productRegistry_;
    bool const primary_;
    Timestamp time_;
  };
}

#endif
