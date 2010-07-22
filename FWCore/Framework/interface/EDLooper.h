#ifndef FWCore_Framework_EDLooper_h
#define FWCore_Framework_EDLooper_h
// -*- C++ -*-
//
// Package:     Framework
// Module:      EDLooper
// 
/**\class EDLooper EDLooper.h FWCore/Framework/interface/EDLooper.h

 Description: Base class for all looping components
 
 This abstract class forms the basis of being able to loop through a list of events multiple times.
 
*/
//
// Author:      Valentin Kuznetsov
// Created:     Wed Jul  5 11:42:17 EDT 2006
// $Id: EDLooper.h,v 1.12 2009/12/21 16:18:27 chrjones Exp $
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

  class EDLooper
  {
    public:

      enum Status {kContinue, kStop};

      EDLooper();
      virtual ~EDLooper();

      void doStartingNewLoop();
      Status doDuringLoop(edm::EventPrincipal& eventPrincipal, const edm::EventSetup& es);
      Status doEndOfLoop(const edm::EventSetup& es);
      void prepareForNextLoop(eventsetup::EventSetupProvider* esp);
      void doBeginRun(RunPrincipal&, EventSetup const&);
      void doEndRun(RunPrincipal &, EventSetup const&);
      void doBeginLuminosityBlock(LuminosityBlockPrincipal &, EventSetup const&);
      void doEndLuminosityBlock(LuminosityBlockPrincipal &, EventSetup const&);


      //This interface is deprecated
      virtual void beginOfJob(const edm::EventSetup&); 
      virtual void beginOfJob();
     
      /**Called before system starts to loop over the events. The argument is a count of
       how many loops have been processed.  For the first time through the events the argument
       will be 0.
       */
      virtual void startingNewLoop(unsigned int ) = 0; 
     
      /**Called after all event modules have had a chance to process the edm::Event.
       */
      virtual Status duringLoop(const edm::Event&, const edm::EventSetup&) = 0; 
     
      /**Called after the system has finished one loop over the events. Thar argument is a 
       count of how many loops have been processed before this loo.  For the first time through
       the events the argument will be 0.
       */
      virtual Status endOfLoop(const edm::EventSetup&, unsigned int iCounter) = 0; 
      virtual void endOfJob();

      ///Called after all event modules have processed the begin of a Run
      virtual void beginRun(Run const&, EventSetup const&);
     
      ///Called after all event modules have processed the end of a Run
      virtual void endRun(Run const&, EventSetup const&);
     
      ///Called after all event modules have processed the begin of a LuminosityBlock
      virtual void beginLuminosityBlock(LuminosityBlock const&, EventSetup const&);

      ///Called after all event modules have processed the end of a LuminosityBlock
      virtual void endLuminosityBlock(LuminosityBlock const&, EventSetup const&);

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

      EDLooper( const EDLooper& ); // stop default
      const EDLooper& operator=( const EDLooper& ); // stop default

      unsigned int iCounter_;
      ActionTable* act_table_;
     
      std::auto_ptr<ScheduleInfo> scheduleInfo_;
      const ModuleChanger* moduleChanger_;     
  };
}

#endif
