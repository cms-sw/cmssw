#ifndef FWCore_ServiceRegistry_ActivityRegistry_h
#define FWCore_ServiceRegistry_ActivityRegistry_h
// -*- C++ -*-
//
// Package:     ServiceRegistry
// Class  :     ActivityRegistry
// 
/**\class ActivityRegistry ActivityRegistry.h FWCore/ServiceRegistry/interface/ActivityRegistry.h

 Description: Registry holding the boost::signals that Services can subscribe to

 Usage:
    Services can connect to the signals distributed by the ActivityRegistry in order to monitor the activity of the application.

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Sep  5 19:53:09 EDT 2005
// $Id: ActivityRegistry.h,v 1.15 2007/02/14 20:45:11 wdd Exp $
//

// system include files
//#include "boost/signal.hpp"
#include "sigc++/signal.h"
#include "boost/bind.hpp"
#include "boost/mem_fn.hpp"

// user include files

#define AR_WATCH_USING_METHOD_0(method) template<class TClass, class TMethod> void method (TClass* iObject, TMethod iMethod) { method (boost::bind(boost::mem_fn(iMethod), iObject)); }
#define AR_WATCH_USING_METHOD_1(method) template<class TClass, class TMethod> void method (TClass* iObject, TMethod iMethod) { method (boost::bind(boost::mem_fn(iMethod), iObject, _1)); }
#define AR_WATCH_USING_METHOD_2(method) template<class TClass, class TMethod> void method (TClass* iObject, TMethod iMethod) { method (boost::bind(boost::mem_fn(iMethod), iObject, _1,_2)); }
// forward declarations
namespace edm {
   class EventID;
   class Timestamp;
   class ModuleDescription;
   class Event;
   class EventSetup;
   class HLTPathStatus;
   
   struct ActivityRegistry
   {
      ActivityRegistry() {}

      // ---------- signals ------------------------------------
      typedef sigc::signal<void> PostBeginJob;
      ///signal is emitted after all modules have gotten their beginJob called
      PostBeginJob postBeginJobSignal_;
      ///convenience function for attaching to signal
      void watchPostBeginJob(const PostBeginJob::slot_type& iSlot) {
         postBeginJobSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_0(watchPostBeginJob)

      typedef sigc::signal<void> PostEndJob;
      ///signal is emitted after all modules have gotten their endJob called
      PostEndJob postEndJobSignal_;
      void watchPostEndJob(const PostEndJob::slot_type& iSlot) {
         PostEndJob::slot_list_type sl = postEndJobSignal_.slots();
         sl.push_front(iSlot);
      }
      AR_WATCH_USING_METHOD_0(watchPostEndJob)

      typedef sigc::signal<void> JobFailure;
      /// signal is emitted if event processing or end-of-job
      /// processing fails with an uncaught exception.
      JobFailure    jobFailureSignal_;
      ///convenience function for attaching to signal
      void watchJobFailure(const JobFailure::slot_type& iSlot) {
         JobFailure::slot_list_type sl = jobFailureSignal_.slots();
         sl.push_front(iSlot);
      }
      AR_WATCH_USING_METHOD_0(watchJobFailure)
      
        /// signal is emitted before the source starts creating the Event
        typedef sigc::signal<void> PreSource;
      PreSource preSourceSignal_;
      void watchPreSource(const PreSource::slot_type& iSlot) {
        preSourceSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_0(watchPreSource)

        /// signal is emitted after the source starts creating the Event
        typedef sigc::signal<void> PostSource;
      PostSource postSourceSignal_;
      void watchPostSource(const PostSource::slot_type& iSlot) {
         PostSource::slot_list_type sl = postSourceSignal_.slots();
         sl.push_front(iSlot);
      }
      AR_WATCH_USING_METHOD_0(watchPostSource)
        
        
        typedef sigc::signal<void, const edm::EventID&, const edm::Timestamp&> PreProcessEvent;
      /// signal is emitted after the Event has been created by the InputSource but before any modules have seen the Event
      PreProcessEvent preProcessEventSignal_;
      void watchPreProcessEvent(const PreProcessEvent::slot_type& iSlot) {
         preProcessEventSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPreProcessEvent)
      
      typedef sigc::signal<void , const Event&, const EventSetup&> PostProcessEvent;
      /// signal is emitted after all modules have finished processing the Event
      PostProcessEvent postProcessEventSignal_;
      void watchPostProcessEvent(const PostProcessEvent::slot_type& iSlot) {
         PostProcessEvent::slot_list_type sl = postProcessEventSignal_.slots();
         sl.push_front(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPostProcessEvent)

      /// signal is emitted before starting to process a Path
      typedef sigc::signal<void, const std::string&> PreProcessPath;
      PreProcessPath preProcessPathSignal_;
      void watchPreProcessPath(const PreProcessPath::slot_type& iSlot) {
        preProcessPathSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPreProcessPath)
        
      /// signal is emitted after all modules have finished for the Path
      typedef sigc::signal<void , const std::string&, const HLTPathStatus&> PostProcessPath;
      PostProcessPath postProcessPathSignal_;
      void watchPostProcessPath(const PostProcessPath::slot_type& iSlot) {
         PostProcessPath::slot_list_type sl = postProcessPathSignal_.slots();
         sl.push_front(iSlot);
      }  
      AR_WATCH_USING_METHOD_2(watchPostProcessPath)
        
      /// signal is emitted before the module is constructed
      typedef sigc::signal<void, const ModuleDescription&> PreModuleConstruction;
      PreModuleConstruction preModuleConstructionSignal_;
      void watchPreModuleConstruction(const PreModuleConstruction::slot_type& iSlot) {
         preModuleConstructionSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPreModuleConstruction)
         
      /// signal is emitted after the module was construction
      typedef sigc::signal<void, const ModuleDescription&> PostModuleConstruction;
      PostModuleConstruction postModuleConstructionSignal_;
      void watchPostModuleConstruction(const PostModuleConstruction::slot_type& iSlot) {
         PostModuleConstruction::slot_list_type sl = postModuleConstructionSignal_.slots();
         sl.push_front(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPostModuleConstruction)

      /// signal is emitted before the module does beginJob
      typedef sigc::signal<void, const ModuleDescription&> PreModuleBeginJob;
      PreModuleBeginJob preModuleBeginJobSignal_;
      void watchPreModuleBeginJob(const PreModuleBeginJob::slot_type& iSlot) {
        preModuleBeginJobSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPreModuleBeginJob)
        
      /// signal is emitted after the module had done beginJob
      typedef sigc::signal<void, const ModuleDescription&> PostModuleBeginJob;
      PostModuleBeginJob postModuleBeginJobSignal_;
      void watchPostModuleBeginJob(const PostModuleBeginJob::slot_type& iSlot) {
         PostModuleBeginJob::slot_list_type sl = postModuleBeginJobSignal_.slots();
         sl.push_front(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPostModuleBeginJob)
        
      /// signal is emitted before the module does endJob
      typedef sigc::signal<void, const ModuleDescription&> PreModuleEndJob;
      PreModuleEndJob preModuleEndJobSignal_;
      void watchPreModuleEndJob(const PreModuleEndJob::slot_type& iSlot) {
        preModuleEndJobSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPreModuleEndJob)
        
      /// signal is emitted after the module had done endJob
      typedef sigc::signal<void, const ModuleDescription&> PostModuleEndJob;
      PostModuleEndJob postModuleEndJobSignal_;
      void watchPostModuleEndJob(const PostModuleEndJob::slot_type& iSlot) {
         PostModuleEndJob::slot_list_type sl = postModuleEndJobSignal_.slots();
         sl.push_front(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPostModuleEndJob)
        
      /// signal is emitted before the module starts processing the Event
      typedef sigc::signal<void, const ModuleDescription&> PreModule;
      PreModule preModuleSignal_;
      void watchPreModule(const PreModule::slot_type& iSlot) {
         preModuleSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPreModule)
         
      /// signal is emitted after the module finished processing the Event
      typedef sigc::signal<void, const ModuleDescription&> PostModule;
      PostModule postModuleSignal_;
      void watchPostModule(const PostModule::slot_type& iSlot) {
         PostModule::slot_list_type sl = postModuleSignal_.slots();
         sl.push_front(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPostModule)
         
        /// signal is emitted before the source is constructed
        typedef sigc::signal<void, const ModuleDescription&> PreSourceConstruction;
      PreSourceConstruction preSourceConstructionSignal_;
      void watchPreSourceConstruction(const PreSourceConstruction::slot_type& iSlot) {
        preSourceConstructionSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPreSourceConstruction)
        
        /// signal is emitted after the source was construction
        typedef sigc::signal<void, const ModuleDescription&> PostSourceConstruction;
      PostSourceConstruction postSourceConstructionSignal_;
      void watchPostSourceConstruction(const PostSourceConstruction::slot_type& iSlot) {
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
      ActivityRegistry(const ActivityRegistry&); // stop default

      const ActivityRegistry& operator=(const ActivityRegistry&); // stop default

      // ---------- member data --------------------------------
      
   };
}
#undef AR_WATCH_USING_METHOD
#endif
