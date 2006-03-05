#ifndef ServiceRegistry_ActivityRegistry_h
#define ServiceRegistry_ActivityRegistry_h
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
// $Id: ActivityRegistry.h,v 1.4 2005/09/28 04:25:02 wmtan Exp $
//

// system include files
#include "boost/signal.hpp"
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
   
   struct ActivityRegistry
   {
      ActivityRegistry() {}

      // ---------- signals ------------------------------------
      typedef boost::signal<void ()> PostBeginJob;
      ///signal is emitted after all modules have gotten their beginJob called
      PostBeginJob postBeginJobSignal_;
      ///convenience function for attaching to signal
      void watchPostBeginJob(const PostBeginJob::slot_type& iSlot) {
         postBeginJobSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_0(watchPostBeginJob)

      typedef boost::signal<void ()> PostEndJob;
      ///signal is emitted after all modules have gotten their endJob called
      PostEndJob postEndJobSignal_;
      void watchPostEndJob(const PostEndJob::slot_type& iSlot) {
         postEndJobSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_0(watchPostEndJob)

      
      typedef boost::signal<void (const edm::EventID&, const edm::Timestamp&)> PreProcessEvent;
      /// signal is emitted after the Event has been created by the InputSource but before any modules have seen the Event
      PreProcessEvent preProcessEventSignal_;
      void watchPreProcessEvent(const PreProcessEvent::slot_type& iSlot) {
         preProcessEventSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPreProcessEvent)
      
      typedef boost::signal<void (const Event&, const EventSetup&)> PostProcessEvent;
      /// signal is emitted after all modules have finished processing the Event
      PostProcessEvent postProcessEventSignal_;
      void watchPostProcessEvent(const PostProcessEvent::slot_type& iSlot) {
         postProcessEventSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPostProcessEvent)

      /// signal is emitted before the module is constructed
      typedef boost::signal<void (const ModuleDescription&)> PreModuleConstruction;
      PreModuleConstruction preModuleConstructionSignal_;
      void watchPreModuleConstruction(const PreModuleConstruction::slot_type& iSlot) {
         preModuleConstructionSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPreModuleConstruction)
         
      /// signal is emitted after the module was construction
      typedef boost::signal<void (const ModuleDescription&)> PostModuleConstruction;
      PostModuleConstruction postModuleConstructionSignal_;
      void watchPostModuleConstruction(const PostModuleConstruction::slot_type& iSlot) {
         postModuleConstructionSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPostModuleConstruction)
         
         /// signal is emitted before the module starts processing the Event
      typedef boost::signal<void (const ModuleDescription&)> PreModule;
      PreModule preModuleSignal_;
      void watchPreModule(const PreModule::slot_type& iSlot) {
         preModuleSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPreModule)
         
      /// signal is emitted after the module finished processing the Event
      typedef boost::signal<void (const ModuleDescription&)> PostModule;
      PostModule postModuleSignal_;
      void watchPostModule(const PostModule::slot_type& iSlot) {
         postModuleSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPostModule)
         
      // ---------- member functions ---------------------------

      ///forwards our signals to slots connected to iOther
      void connect(ActivityRegistry& iOther);
      
   private:
      ActivityRegistry(const ActivityRegistry&); // stop default

      const ActivityRegistry& operator=(const ActivityRegistry&); // stop default

      // ---------- member data --------------------------------
      
   };
}
#undef AR_WATCH_USING_METHOD
#endif
