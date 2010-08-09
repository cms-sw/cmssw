#ifndef FWCore_Framework_EDLooperBase_h
#define FWCore_Framework_EDLooperBase_h
// -*- C++ -*-
//
// Package:     Framework
// Module:      EDLooperBase
// 
/**\class EDLooperBase EDLooperBase.h FWCore/Framework/interface/EDLooperBase.h

 Description: Base class for all looping components
 
 This abstract class forms the basis of being able to loop through a list of events multiple times. In general the
 class EDLooper is more appropriate for looping sequentially over events.
 
 The class uses a Template Pattern to describe the structure of a loop.  The structure is made up of three phases
 1) start of loop
   At the start of a new loop the virtual method 'startingNewLoop(unsigned int)' is called.  The integer passed is
   the number of loops which have been run in the job starting at index 0.
 2) during loop
   Each time an event has been processed by all other modules in a job, the method 
      Status duringLoop(const edm::Event&, const edm::EventSetup&, ProcessingController&)
   will be called.  The return value 'Status' can be either kContinue if you want to proceed to the 'next' event or
   kStop if you want to stop this particular loop.  
   The class ProcessingController allows you further control of what is meant by 'next' event.  By default 'next' event
   will just be the next event in the normal sequence.  However, for sources which support the feature, you can also use
   ProcessingController to make the 'next' event be the 'event before the last processed event' (i.e. go back one) or to
   use an edm::EventID to exactly specify what next event you want.  However, if you say to go 'back' one event while the 
   job is on the first event of the loop or pass an edm::EventID for an event not contained in the source the job will
   immediately go to 'end of loop'.
   NOTE: if you have no need of controlling exactly what 'next' event should be processed then you should instead inherit
   from the subclass EDLooper and us it simplified interface for 'duringLoop'.
 3) end of loop
   Once all events have been processed or kStop was returned from 'duringLoop' or ProcessingController was told to go to an
   'invalid' event then the method
      Status endOfLoop(const edm::EventSetup&, unsigned int iCounter)
   will be called.  iCounter will be the number of loops which have been run in the job starting at index 0. If kContinue
   is returned from endOfLoop then a new loop will begin else if kStop is called then the job will end.
 
 Like other modules, an EDLooperBase is called for 'beginJob', 'endJob', 'beginRun', 'endRun', 'beginLuminosityBlock' and 
 'endLuminosityBlock'.
 
 Additional information and control of a job is possible via the interfaces:
 attachTo(edm::ActivityRegistry&) : via the ActivityRegistry you can monitor exactly which modules are being run
 scheduleInfo(): returns a ScheduleInfo which you can use to determine what paths are in a job and what 
    modules are on each path.
 moduleChanger(): returns a ModuleChanger instance which can be used to modify the parameters of an EDLooper or EDFilter.
    Such modifications can only occur during a call to 'endOfLoop' since the newly changed module can only be properly initialized
    at the start of the next loop.
 
*/
//
// Author:      Chris Jones
// Created:     Mon Aug  9 12:42:17 EDT 2010
// $Id: EDLooperBase.h,v 1.13 2010/07/22 15:00:27 chrjones Exp $
//

#include "FWCore/Framework/interface/Frameworkfwd.h"

#include <set>
#include <memory>

namespace edm {
  namespace eventsetup {
    class EventSetupRecordKey;
    class EventSetupProvider;
  }
  class ActionTable;
  class ScheduleInfo;
  class ModuleChanger;
  class ProcessingController;
  class ActivityRegistry;

  class EDLooperBase
  {
    public:

      enum Status {kContinue, kStop};

      EDLooperBase();
      virtual ~EDLooperBase();

      void doStartingNewLoop();
      Status doDuringLoop(edm::EventPrincipal& eventPrincipal, const edm::EventSetup& es, edm::ProcessingController&);
      Status doEndOfLoop(const edm::EventSetup& es);
      void prepareForNextLoop(eventsetup::EventSetupProvider* esp);
      void doBeginRun(RunPrincipal&, EventSetup const&);
      void doEndRun(RunPrincipal &, EventSetup const&);
      void doBeginLuminosityBlock(LuminosityBlockPrincipal &, EventSetup const&);
      void doEndLuminosityBlock(LuminosityBlockPrincipal &, EventSetup const&);


      //This interface is deprecated
      virtual void beginOfJob(const edm::EventSetup&); 
      virtual void beginOfJob();
     
      virtual void endOfJob();

      ///Override this method if you need to monitor the state of the processing
      virtual void attachTo(ActivityRegistry&);
                       
      void setActionTable(ActionTable* actionTable) { act_table_ = actionTable; }

      virtual std::set<eventsetup::EventSetupRecordKey> modifyingRecords() const;
     
      void copyInfo(const ScheduleInfo&);
      void setModuleChanger(const ModuleChanger*);

    protected:
      ///This only returns a non-zero value during the call to endOfLoop
      const ModuleChanger* moduleChanger() const;
      ///This returns a non-zero value after the constructor has been called
      const ScheduleInfo* scheduleInfo() const;
    private:

      EDLooperBase( const EDLooperBase& ); // stop default
      const EDLooperBase& operator=( const EDLooperBase& ); // stop default

      /**Called before system starts to loop over the events. The argument is a count of
       how many loops have been processed.  For the first time through the events the argument
       will be 0.
       */
      virtual void startingNewLoop(unsigned int ) = 0; 
    
      /**Called after all event modules have had a chance to process the edm::Event.
       */
      virtual Status duringLoop(const edm::Event&, const edm::EventSetup&, ProcessingController&) = 0; 
    
      /**Called after the system has finished one loop over the events. Thar argument is a 
       count of how many loops have been processed before this loo.  For the first time through
       the events the argument will be 0.
       */
      virtual Status endOfLoop(const edm::EventSetup&, unsigned int iCounter) = 0; 

      ///Called after all event modules have processed the begin of a Run
      virtual void beginRun(Run const&, EventSetup const&);
    
      ///Called after all event modules have processed the end of a Run
      virtual void endRun(Run const&, EventSetup const&);
    
      ///Called after all event modules have processed the begin of a LuminosityBlock
      virtual void beginLuminosityBlock(LuminosityBlock const&, EventSetup const&);
    
      ///Called after all event modules have processed the end of a LuminosityBlock
      virtual void endLuminosityBlock(LuminosityBlock const&, EventSetup const&);
    
    
      unsigned int iCounter_;
      ActionTable* act_table_;
     
      std::auto_ptr<ScheduleInfo> scheduleInfo_;
      const ModuleChanger* moduleChanger_;     
  };
}

#endif
