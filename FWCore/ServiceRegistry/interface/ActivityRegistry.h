#ifndef FWCore_ServiceRegistry_ActivityRegistry_h
#define FWCore_ServiceRegistry_ActivityRegistry_h
// -*- C++ -*-
//
// Package:     ServiceRegistry
// Class  :     ActivityRegistry
// 
/**\class ActivityRegistry ActivityRegistry.h FWCore/ServiceRegistry/interface/ActivityRegistry.h

 Description: Registry holding the signals that Services can subscribe to

 Usage:
    Services can connect to the signals distributed by the ActivityRegistry in order to monitor the activity of the application.

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Sep  5 19:53:09 EDT 2005
// $Id: ActivityRegistry.h,v 1.16 2007/06/14 02:25:50 wmtan Exp $
//

// system include files
//#include "boost/signal.hpp"
#include "sigc++/signal.h"
#include "boost/bind.hpp"
#include "boost/mem_fn.hpp"
#include "boost/utility.hpp"

// user include files

#define AR_WATCH_USING_METHOD_0(method) template<class TClass, class TMethod> void method (TClass* iObject, TMethod iMethod) { method (boost::bind(boost::mem_fn(iMethod), iObject)); }
#define AR_WATCH_USING_METHOD_1(method) template<class TClass, class TMethod> void method (TClass* iObject, TMethod iMethod) { method (boost::bind(boost::mem_fn(iMethod), iObject, _1)); }
#define AR_WATCH_USING_METHOD_2(method) template<class TClass, class TMethod> void method (TClass* iObject, TMethod iMethod) { method (boost::bind(boost::mem_fn(iMethod), iObject, _1,_2)); }
// forward declarations
namespace edm {
   class EventID;
   class LuminosityBlockID;
   class RunID;
   class Timestamp;
   class ModuleDescription;
   class Event;
   class LuminosityBlock;
   class Run;
   class EventSetup;
   class HLTPathStatus;
   
   struct ActivityRegistry : private boost::noncopyable
   {
      ActivityRegistry() {}

      // ---------- signals ------------------------------------
      typedef sigc::signal<void> PostBeginJob;
      ///signal is emitted after all modules have gotten their beginJob called
      PostBeginJob postBeginJobSignal_;
      ///convenience function for attaching to signal
      void watchPostBeginJob(PostBeginJob::slot_type const& iSlot) {
         postBeginJobSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_0(watchPostBeginJob)

      typedef sigc::signal<void> PostEndJob;
      ///signal is emitted after all modules have gotten their endJob called
      PostEndJob postEndJobSignal_;
      void watchPostEndJob(PostEndJob::slot_type const& iSlot) {
         PostEndJob::slot_list_type sl = postEndJobSignal_.slots();
         sl.push_front(iSlot);
      }
      AR_WATCH_USING_METHOD_0(watchPostEndJob)

      typedef sigc::signal<void> JobFailure;
      /// signal is emitted if event processing or end-of-job
      /// processing fails with an uncaught exception.
      JobFailure    jobFailureSignal_;
      ///convenience function for attaching to signal
      void watchJobFailure(JobFailure::slot_type const& iSlot) {
         JobFailure::slot_list_type sl = jobFailureSignal_.slots();
         sl.push_front(iSlot);
      }
      AR_WATCH_USING_METHOD_0(watchJobFailure)
      
      /// signal is emitted before the source starts creating an Event
      typedef sigc::signal<void> PreSource;
      PreSource preSourceSignal_;
      void watchPreSource(PreSource::slot_type const& iSlot) {
        preSourceSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_0(watchPreSource)

      /// signal is emitted after the source starts creating an Event
      typedef sigc::signal<void> PostSource;
      PostSource postSourceSignal_;
      void watchPostSource(PostSource::slot_type const& iSlot) {
         PostSource::slot_list_type sl = postSourceSignal_.slots();
         sl.push_front(iSlot);
      }
      AR_WATCH_USING_METHOD_0(watchPostSource)
        
      /// signal is emitted before the source starts creating a Lumi
      typedef sigc::signal<void> PreSourceLumi;
      PreSourceLumi preSourceLumiSignal_;
      void watchPreSourceLumi(PreSourceLumi::slot_type const& iSlot) {
        preSourceLumiSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_0(watchPreSourceLumi)

      /// signal is emitted after the source starts creating a Lumi
      typedef sigc::signal<void> PostSourceLumi;
      PostSourceLumi postSourceLumiSignal_;
      void watchPostSourceLumi(PostSourceLumi::slot_type const& iSlot) {
         PostSourceLumi::slot_list_type sl = postSourceLumiSignal_.slots();
         sl.push_front(iSlot);
      }
      AR_WATCH_USING_METHOD_0(watchPostSourceLumi)
        
      /// signal is emitted before the source starts creating a Run
      typedef sigc::signal<void> PreSourceRun;
      PreSourceRun preSourceRunSignal_;
      void watchPreSourceRun(PreSourceRun::slot_type const& iSlot) {
        preSourceRunSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_0(watchPreSourceRun)

      /// signal is emitted after the source starts creating a Run
      typedef sigc::signal<void> PostSourceRun;
      PostSourceRun postSourceRunSignal_;
      void watchPostSourceRun(PostSourceRun::slot_type const& iSlot) {
         PostSourceRun::slot_list_type sl = postSourceRunSignal_.slots();
         sl.push_front(iSlot);
      }
      AR_WATCH_USING_METHOD_0(watchPostSourceRun)
        
      /// signal is emitted before the source opens a file
      typedef sigc::signal<void> PreSourceFile;
      PreSourceFile preSourceFileSignal_;
      void watchPreSourceFile(PreSourceFile::slot_type const& iSlot) {
        preSourceFileSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_0(watchPreSourceFile)

      /// signal is emitted after the source opens a file
      typedef sigc::signal<void> PostSourceFile;
      PostSourceFile postSourceFileSignal_;
      void watchPostSourceFile(PostSourceFile::slot_type const& iSlot) {
         PostSourceFile::slot_list_type sl = postSourceFileSignal_.slots();
         sl.push_front(iSlot);
      }
      AR_WATCH_USING_METHOD_0(watchPostSourceFile)
        
      typedef sigc::signal<void, edm::EventID const&, edm::Timestamp const&> PreProcessEvent;
      /// signal is emitted after the Event has been created by the InputSource but before any modules have seen the Event
      PreProcessEvent preProcessEventSignal_;
      void watchPreProcessEvent(PreProcessEvent::slot_type const& iSlot) {
         preProcessEventSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPreProcessEvent)
      
      typedef sigc::signal<void, Event const&, EventSetup const&> PostProcessEvent;
      /// signal is emitted after all modules have finished processing the Event
      PostProcessEvent postProcessEventSignal_;
      void watchPostProcessEvent(PostProcessEvent::slot_type const& iSlot) {
         PostProcessEvent::slot_list_type sl = postProcessEventSignal_.slots();
         sl.push_front(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPostProcessEvent)

      typedef sigc::signal<void, edm::RunID const&, edm::Timestamp const&> PreBeginRun;
      /// signal is emitted after the Run has been created by the InputSource but before any modules have seen the Run
      PreBeginRun preBeginRunSignal_;
      void watchPreBeginRun(PreBeginRun::slot_type const& iSlot) {
         preBeginRunSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPreBeginRun)
      
      typedef sigc::signal<void, Run const&, EventSetup const&> PostBeginRun;
      /// signal is emitted after all modules have finished processing the beginRun 
      PostBeginRun postBeginRunSignal_;
      void watchPostBeginRun(PostBeginRun::slot_type const& iSlot) {
         PostBeginRun::slot_list_type sl = postBeginRunSignal_.slots();
         sl.push_front(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPostBeginRun)

      typedef sigc::signal<void, edm::RunID const&, edm::Timestamp const&> PreEndRun;
      /// signal is emitted before the endRun is processed
      PreEndRun preEndRunSignal_;
      void watchPreEndRun(PreEndRun::slot_type const& iSlot) {
         preEndRunSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPreEndRun)
      
      typedef sigc::signal<void, Run const&, EventSetup const&> PostEndRun;
      /// signal is emitted after all modules have finished processing the Run
      PostEndRun postEndRunSignal_;
      void watchPostEndRun(PostEndRun::slot_type const& iSlot) {
         PostEndRun::slot_list_type sl = postEndRunSignal_.slots();
         sl.push_front(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPostEndRun)

      typedef sigc::signal<void, edm::LuminosityBlockID const&, edm::Timestamp const&> PreBeginLumi;
      /// signal is emitted after the Lumi has been created by the InputSource but before any modules have seen the Lumi
      PreBeginLumi preBeginLumiSignal_;
      void watchPreBeginLumi(PreBeginLumi::slot_type const& iSlot) {
         preBeginLumiSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPreBeginLumi)
      
      typedef sigc::signal<void, LuminosityBlock const&, EventSetup const&> PostBeginLumi;
      /// signal is emitted after all modules have finished processing the beginLumi
      PostBeginLumi postBeginLumiSignal_;
      void watchPostBeginLumi(PostBeginLumi::slot_type const& iSlot) {
         PostBeginLumi::slot_list_type sl = postBeginLumiSignal_.slots();
         sl.push_front(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPostBeginLumi)

      typedef sigc::signal<void, edm::LuminosityBlockID const&, edm::Timestamp const&> PreEndLumi;
      /// signal is emitted before the endLumi is processed
      PreEndLumi preEndLumiSignal_;
      void watchPreEndLumi(PreEndLumi::slot_type const& iSlot) {
         preEndLumiSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPreEndLumi)
      
      typedef sigc::signal<void, LuminosityBlock const&, EventSetup const&> PostEndLumi;
      /// signal is emitted after all modules have finished processing the Lumi
      PostEndLumi postEndLumiSignal_;
      void watchPostEndLumi(PostEndLumi::slot_type const& iSlot) {
         PostEndLumi::slot_list_type sl = postEndLumiSignal_.slots();
         sl.push_front(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPostEndLumi)

      /// signal is emitted before starting to process a Path for an event
      typedef sigc::signal<void, std::string const&> PreProcessPath;
      PreProcessPath preProcessPathSignal_;
      void watchPreProcessPath(PreProcessPath::slot_type const& iSlot) {
        preProcessPathSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPreProcessPath)
        
      /// signal is emitted after all modules have finished for the Path for an event
      typedef sigc::signal<void, std::string const&, HLTPathStatus const&> PostProcessPath;
      PostProcessPath postProcessPathSignal_;
      void watchPostProcessPath(PostProcessPath::slot_type const& iSlot) {
         PostProcessPath::slot_list_type sl = postProcessPathSignal_.slots();
         sl.push_front(iSlot);
      }  
      AR_WATCH_USING_METHOD_2(watchPostProcessPath)
        
      /// signal is emitted before starting to process a Path for beginRun
      typedef sigc::signal<void, std::string const&> PrePathBeginRun;
      PrePathBeginRun prePathBeginRunSignal_;
      void watchPrePathBeginRun(PrePathBeginRun::slot_type const& iSlot) {
        prePathBeginRunSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPrePathBeginRun)
        
      /// signal is emitted after all modules have finished for the Path for beginRun
      typedef sigc::signal<void, std::string const&, HLTPathStatus const&> PostPathBeginRun;
      PostPathBeginRun postPathBeginRunSignal_;
      void watchPostPathBeginRun(PostPathBeginRun::slot_type const& iSlot) {
         PostPathBeginRun::slot_list_type sl = postPathBeginRunSignal_.slots();
         sl.push_front(iSlot);
      }  
      AR_WATCH_USING_METHOD_2(watchPostPathBeginRun)
        
      /// signal is emitted before starting to process a Path for endRun
      typedef sigc::signal<void, std::string const&> PrePathEndRun;
      PrePathEndRun prePathEndRunSignal_;
      void watchPrePathEndRun(PrePathEndRun::slot_type const& iSlot) {
        prePathEndRunSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPrePathEndRun)
        
      /// signal is emitted after all modules have finished for the Path for endRun
      typedef sigc::signal<void, std::string const&, HLTPathStatus const&> PostPathEndRun;
      PostPathEndRun postPathEndRunSignal_;
      void watchPostPathEndRun(PostPathEndRun::slot_type const& iSlot) {
         PostPathEndRun::slot_list_type sl = postPathEndRunSignal_.slots();
         sl.push_front(iSlot);
      }  
      AR_WATCH_USING_METHOD_2(watchPostPathEndRun)
        
      /// signal is emitted before starting to process a Path for beginLumi
      typedef sigc::signal<void, std::string const&> PrePathBeginLumi;
      PrePathBeginLumi prePathBeginLumiSignal_;
      void watchPrePathBeginLumi(PrePathBeginLumi::slot_type const& iSlot) {
        prePathBeginLumiSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPrePathBeginLumi)
        
      /// signal is emitted after all modules have finished for the Path for beginLumi
      typedef sigc::signal<void, std::string const&, HLTPathStatus const&> PostPathBeginLumi;
      PostPathBeginLumi postPathBeginLumiSignal_;
      void watchPostPathBeginLumi(PostPathBeginLumi::slot_type const& iSlot) {
         PostPathBeginLumi::slot_list_type sl = postPathBeginLumiSignal_.slots();
         sl.push_front(iSlot);
      }  
      AR_WATCH_USING_METHOD_2(watchPostPathBeginLumi)
        
      /// signal is emitted before starting to process a Path for endRun
      typedef sigc::signal<void, std::string const&> PrePathEndLumi;
      PrePathEndLumi prePathEndLumiSignal_;
      void watchPrePathEndLumi(PrePathEndLumi::slot_type const& iSlot) {
        prePathEndLumiSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPrePathEndLumi)
        
      /// signal is emitted after all modules have finished for the Path for endRun
      typedef sigc::signal<void, std::string const&, HLTPathStatus const&> PostPathEndLumi;
      PostPathEndLumi postPathEndLumiSignal_;
      void watchPostPathEndLumi(PostPathEndLumi::slot_type const& iSlot) {
         PostPathEndLumi::slot_list_type sl = postPathEndLumiSignal_.slots();
         sl.push_front(iSlot);
      }  
      AR_WATCH_USING_METHOD_2(watchPostPathEndLumi)
        
      /// signal is emitted before the module is constructed
      typedef sigc::signal<void, ModuleDescription const&> PreModuleConstruction;
      PreModuleConstruction preModuleConstructionSignal_;
      void watchPreModuleConstruction(PreModuleConstruction::slot_type const& iSlot) {
         preModuleConstructionSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPreModuleConstruction)
         
      /// signal is emitted after the module was construction
      typedef sigc::signal<void, ModuleDescription const&> PostModuleConstruction;
      PostModuleConstruction postModuleConstructionSignal_;
      void watchPostModuleConstruction(PostModuleConstruction::slot_type const& iSlot) {
         PostModuleConstruction::slot_list_type sl = postModuleConstructionSignal_.slots();
         sl.push_front(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPostModuleConstruction)

      /// signal is emitted before the module does beginJob
      typedef sigc::signal<void, ModuleDescription const&> PreModuleBeginJob;
      PreModuleBeginJob preModuleBeginJobSignal_;
      void watchPreModuleBeginJob(PreModuleBeginJob::slot_type const& iSlot) {
        preModuleBeginJobSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPreModuleBeginJob)
        
      /// signal is emitted after the module had done beginJob
      typedef sigc::signal<void, ModuleDescription const&> PostModuleBeginJob;
      PostModuleBeginJob postModuleBeginJobSignal_;
      void watchPostModuleBeginJob(PostModuleBeginJob::slot_type const& iSlot) {
         PostModuleBeginJob::slot_list_type sl = postModuleBeginJobSignal_.slots();
         sl.push_front(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPostModuleBeginJob)
        
      /// signal is emitted before the module does endJob
      typedef sigc::signal<void, ModuleDescription const&> PreModuleEndJob;
      PreModuleEndJob preModuleEndJobSignal_;
      void watchPreModuleEndJob(PreModuleEndJob::slot_type const& iSlot) {
        preModuleEndJobSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPreModuleEndJob)
        
      /// signal is emitted after the module had done endJob
      typedef sigc::signal<void, ModuleDescription const&> PostModuleEndJob;
      PostModuleEndJob postModuleEndJobSignal_;
      void watchPostModuleEndJob(PostModuleEndJob::slot_type const& iSlot) {
         PostModuleEndJob::slot_list_type sl = postModuleEndJobSignal_.slots();
         sl.push_front(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPostModuleEndJob)
        
      /// signal is emitted before the module starts processing the Event
      typedef sigc::signal<void, ModuleDescription const&> PreModule;
      PreModule preModuleSignal_;
      void watchPreModule(PreModule::slot_type const& iSlot) {
         preModuleSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPreModule)
         
      /// signal is emitted after the module finished processing the Event
      typedef sigc::signal<void, ModuleDescription const&> PostModule;
      PostModule postModuleSignal_;
      void watchPostModule(PostModule::slot_type const& iSlot) {
         PostModule::slot_list_type sl = postModuleSignal_.slots();
         sl.push_front(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPostModule)
         
      /// signal is emitted before the module starts processing beginRun
      typedef sigc::signal<void, ModuleDescription const&> PreModuleBeginRun;
      PreModuleBeginRun preModuleBeginRunSignal_;
      void watchPreModuleBeginRun(PreModuleBeginRun::slot_type const& iSlot) {
         preModuleBeginRunSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPreModuleBeginRun)
         
      /// signal is emitted after the module finished processing beginRun
      typedef sigc::signal<void, ModuleDescription const&> PostModuleBeginRun;
      PostModuleBeginRun postModuleBeginRunSignal_;
      void watchPostModuleBeginRun(PostModuleBeginRun::slot_type const& iSlot) {
         PostModuleBeginRun::slot_list_type sl = postModuleBeginRunSignal_.slots();
         sl.push_front(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPostModuleBeginRun)
         
      /// signal is emitted before the module starts processing endRun
      typedef sigc::signal<void, ModuleDescription const&> PreModuleEndRun;
      PreModuleEndRun preModuleEndRunSignal_;
      void watchPreModuleEndRun(PreModuleEndRun::slot_type const& iSlot) {
         preModuleEndRunSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPreModuleEndRun)
         
      /// signal is emitted after the module finished processing endRun
      typedef sigc::signal<void, ModuleDescription const&> PostModuleEndRun;
      PostModuleEndRun postModuleEndRunSignal_;
      void watchPostModuleEndRun(PostModuleEndRun::slot_type const& iSlot) {
         PostModuleEndRun::slot_list_type sl = postModuleEndRunSignal_.slots();
         sl.push_front(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPostModuleEndRun)
         
      /// signal is emitted before the module starts processing beginLumi
      typedef sigc::signal<void, ModuleDescription const&> PreModuleBeginLumi;
      PreModuleBeginLumi preModuleBeginLumiSignal_;
      void watchPreModuleBeginLumi(PreModuleBeginLumi::slot_type const& iSlot) {
         preModuleBeginLumiSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPreModuleBeginLumi)
         
      /// signal is emitted after the module finished processing beginLumi
      typedef sigc::signal<void, ModuleDescription const&> PostModuleBeginLumi;
      PostModuleBeginLumi postModuleBeginLumiSignal_;
      void watchPostModuleBeginLumi(PostModuleBeginLumi::slot_type const& iSlot) {
         PostModuleBeginLumi::slot_list_type sl = postModuleBeginLumiSignal_.slots();
         sl.push_front(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPostModuleBeginLumi)
         
      /// signal is emitted before the module starts processing endLumi
      typedef sigc::signal<void, ModuleDescription const&> PreModuleEndLumi;
      PreModuleEndLumi preModuleEndLumiSignal_;
      void watchPreModuleEndLumi(PreModuleEndLumi::slot_type const& iSlot) {
         preModuleEndLumiSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPreModuleEndLumi)
         
      /// signal is emitted after the module finished processing endLumi
      typedef sigc::signal<void, ModuleDescription const&> PostModuleEndLumi;
      PostModuleEndLumi postModuleEndLumiSignal_;
      void watchPostModuleEndLumi(PostModuleEndLumi::slot_type const& iSlot) {
         PostModuleEndLumi::slot_list_type sl = postModuleEndLumiSignal_.slots();
         sl.push_front(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPostModuleEndLumi)
         
      /// signal is emitted before the source is constructed
      typedef sigc::signal<void, ModuleDescription const&> PreSourceConstruction;
      PreSourceConstruction preSourceConstructionSignal_;
      void watchPreSourceConstruction(PreSourceConstruction::slot_type const& iSlot) {
        preSourceConstructionSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPreSourceConstruction)
        
      /// signal is emitted after the source was construction
      typedef sigc::signal<void, ModuleDescription const&> PostSourceConstruction;
      PostSourceConstruction postSourceConstructionSignal_;
      void watchPostSourceConstruction(PostSourceConstruction::slot_type const& iSlot) {
         PostSourceConstruction::slot_list_type sl = postSourceConstructionSignal_.slots();
         sl.push_front(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPostSourceConstruction)
        // ---------- member functions ---------------------------

      ///forwards our signals to slots connected to iOther
      void connect(ActivityRegistry& iOther);
      
      ///copy the slots from iOther and connect them directly to our own
      /// this allows us to 'forward' signals more efficiently,
      /// BUT if iOther gains new slots after this call, we will not see them
      /// This is also careful to keep the order of the slots proper
      /// for services.
      void copySlotsFrom(ActivityRegistry& iOther);
      
   private:
   };
}
#undef AR_WATCH_USING_METHOD
#endif
