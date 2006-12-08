#ifndef Framework_InputSource_h
#define Framework_InputSource_h


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

$Id: InputSource.h,v 1.16 2006/11/09 23:12:59 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>
#include <string>

#include "DataFormats/Common/interface/EventID.h"
#include "DataFormats/Common/interface/EDProductfwd.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Framework/interface/ProductRegistryHelper.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

namespace edm {
  class ParameterSet;

  class InputSource : private ProductRegistryHelper {
  public:
    typedef ProductRegistryHelper::TypeLabelList TypeLabelList;
    /// Constructor
    explicit InputSource(ParameterSet const&, InputSourceDescription const&);

    /// Destructor
    virtual ~InputSource();

    /// Indicate inability to get a new event by returning a null auto_ptr.

    /// Read next event
    std::auto_ptr<EventPrincipal> readEvent();

    /// Read a specific event
    std::auto_ptr<EventPrincipal> readEvent(EventID const&);

    /// Skip the number of events specified.
    /// Offset may be negative.
    void skipEvents(int offset);

    /// Begin again at the first event
    void rewind() {rewind_();}

    /// Wake up the input source
    void wakeUp() {wakeUp_();}

    /// Set the run number
    void setRunNumber(RunNumber_t r) {setRun(r);}

    /// issue an event report
    void issueReports(EventID const&);

    /// Register any produced products
    void registerProducts();

    /// Accessor for product registry.
    ProductRegistry & productRegistry() const {return *isDesc_.productRegistry_;}
    
    /// Reset the remaining number of events to the maximum number.
    void repeat() {remainingEvents_ = maxEvents_;}

    /// Accessor for maximum number of events to be read.
    /// -1 is used for unlimited.
    int maxEvents() const {return maxEvents_;}

    /// Accessor for remaining number of events to be read.
    int remainingEvents() const {return remainingEvents_;}

    /// Accessor for 'module' description.
    ModuleDescription const& moduleDescription() const {return isDesc_.moduleDescription_;}

    /// Accessor fpr primary input source flag
    bool const primary() const {return primary_;}

    /// Called by framework at beginning of job
    void doBeginJob(EventSetup const&);

    /// Called by framework at end of job
    void doEndJob();

    using ProductRegistryHelper::produces;
    using ProductRegistryHelper::typeLabelList;

  private:

    // Indicate inability to get a new event by returning a null
    // auto_ptr.
    std::auto_ptr<EventPrincipal> readEvent_();

    std::auto_ptr<EventPrincipal> readEvent_(EventID const&);

    virtual std::auto_ptr<EventPrincipal> read() = 0;

    virtual std::auto_ptr<EventPrincipal> readIt(EventID const&);

    virtual void skip(int);

    virtual void setRun(RunNumber_t r);

    virtual void rewind_();

    virtual void wakeUp_(){}

    void preRead();

    void postRead(Event& event);

    virtual void beginJob(EventSetup const&){}

    virtual void endJob(){}

  private:

    int const maxEvents_;

    int remainingEvents_;

    int readCount_;

    bool const unlimited_;

    InputSourceDescription isDesc_;

    bool const primary_;
  };
}

#endif
