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

There are unit tests for the signals that use the Tracer
to print out the transitions as they occur and then
compare to a reference file. One test does this for
a SubProcess test and the other for a test using
unscheduled execution. The tests are in FWCore/Integration/test:
  run_SubProcess.sh
  testSubProcess_cfg.py
  run_TestGetBy.sh
  testGetBy1_cfg.py
  testGetBy2_cfg.py
*/
//
// Original Author:  Chris Jones
//         Created:  Mon Sep  5 19:53:09 EDT 2005
//

// system include files
//#include "boost/signal.hpp"
#include "FWCore/Utilities/interface/Signal.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "boost/bind.hpp"
#include "boost/mem_fn.hpp"
#include "boost/utility.hpp"

// user include files

#define AR_WATCH_USING_METHOD_0(method) template<class TClass, class TMethod> void method (TClass* iObject, TMethod iMethod) { method (boost::bind(boost::mem_fn(iMethod), iObject)); }
#define AR_WATCH_USING_METHOD_1(method) template<class TClass, class TMethod> void method (TClass* iObject, TMethod iMethod) { method (boost::bind(boost::mem_fn(iMethod), iObject, _1)); }
#define AR_WATCH_USING_METHOD_2(method) template<class TClass, class TMethod> void method (TClass* iObject, TMethod iMethod) { method (boost::bind(boost::mem_fn(iMethod), iObject, _1, _2)); }
#define AR_WATCH_USING_METHOD_3(method) template<class TClass, class TMethod> void method (TClass* iObject, TMethod iMethod) { method (boost::bind(boost::mem_fn(iMethod), iObject, _1, _2, _3)); }
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
   class GlobalContext;
   class StreamContext;
   class PathContext;
   class ModuleCallingContext;
   namespace service {
     class SystemBounds;
   }
  
   namespace signalslot {
      void throwObsoleteSignalException();
      
      template<class T>
      class ObsoleteSignal {
      public:
         typedef std::function<T> slot_type;

         ObsoleteSignal() = default;

         template<typename... Args>
         void emit(Args&&... args) const {}
         
         template<typename... Args>
         void operator()(Args&&... args) const {}

         template<typename U>
         void connect(U iFunc) {
            throwObsoleteSignalException();
         }
         
         template<typename U>
         void connect_front(U iFunc) {
            throwObsoleteSignalException();
         }

      };
   }
   class ActivityRegistry : private boost::noncopyable {
   public:
      ActivityRegistry() {}

      // ---------- signals ------------------------------------
      typedef signalslot::Signal<void(service::SystemBounds const&)> Preallocate;
      ///signal is emitted before beginJob
      Preallocate preallocateSignal_;
      void watchPreallocate(Preallocate::slot_type const& iSlot) {
        preallocateSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPreallocate)
     
      typedef signalslot::Signal<void()> PostBeginJob;
      ///signal is emitted after all modules have gotten their beginJob called
      PostBeginJob postBeginJobSignal_;
      ///convenience function for attaching to signal
      void watchPostBeginJob(PostBeginJob::slot_type const& iSlot) {
         postBeginJobSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_0(watchPostBeginJob)

      typedef signalslot::Signal<void()> PostEndJob;
      ///signal is emitted after all modules have gotten their endJob called
      PostEndJob postEndJobSignal_;
      void watchPostEndJob(PostEndJob::slot_type const& iSlot) {
         postEndJobSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_0(watchPostEndJob)

      typedef signalslot::Signal<void()> JobFailure;
      /// signal is emitted if event processing or end-of-job
      /// processing fails with an uncaught exception.
      JobFailure    jobFailureSignal_;
      ///convenience function for attaching to signal
      void watchJobFailure(JobFailure::slot_type const& iSlot) {
         jobFailureSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_0(watchJobFailure)
      
      /// signal is emitted before the source starts creating an Event
      typedef signalslot::Signal<void(StreamID)> PreSourceEvent;
      PreSourceEvent preSourceSignal_;
      void watchPreSourceEvent(PreSourceEvent::slot_type const& iSlot) {
        preSourceSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPreSourceEvent)

      /// signal is emitted after the source starts creating an Event
      typedef signalslot::Signal<void(StreamID)> PostSourceEvent;
      PostSourceEvent postSourceSignal_;
      void watchPostSourceEvent(PostSourceEvent::slot_type const& iSlot) {
         postSourceSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPostSourceEvent)
        
      /// signal is emitted before the source starts creating a Lumi
      typedef signalslot::Signal<void()> PreSourceLumi;
      PreSourceLumi preSourceLumiSignal_;
      void watchPreSourceLumi(PreSourceLumi::slot_type const& iSlot) {
        preSourceLumiSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_0(watchPreSourceLumi)

      /// signal is emitted after the source starts creating a Lumi
      typedef signalslot::Signal<void()> PostSourceLumi;
      PostSourceLumi postSourceLumiSignal_;
      void watchPostSourceLumi(PostSourceLumi::slot_type const& iSlot) {
         postSourceLumiSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_0(watchPostSourceLumi)
        
      /// signal is emitted before the source starts creating a Run
      typedef signalslot::Signal<void()> PreSourceRun;
      PreSourceRun preSourceRunSignal_;
      void watchPreSourceRun(PreSourceRun::slot_type const& iSlot) {
        preSourceRunSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_0(watchPreSourceRun)

      /// signal is emitted after the source starts creating a Run
      typedef signalslot::Signal<void()> PostSourceRun;
      PostSourceRun postSourceRunSignal_;
      void watchPostSourceRun(PostSourceRun::slot_type const& iSlot) {
         postSourceRunSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_0(watchPostSourceRun)
        
      /// signal is emitted before the source opens a file
      typedef signalslot::Signal<void(std::string const&, bool)> PreOpenFile;
      PreOpenFile preOpenFileSignal_;
      void watchPreOpenFile(PreOpenFile::slot_type const& iSlot) {
        preOpenFileSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPreOpenFile)

      /// signal is emitted after the source opens a file
      //   Note this is only done for a primary file, not a secondary one.
      typedef signalslot::Signal<void(std::string const&, bool)> PostOpenFile;
      PostOpenFile postOpenFileSignal_;
      void watchPostOpenFile(PostOpenFile::slot_type const& iSlot) {
         postOpenFileSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPostOpenFile)
        
      /// signal is emitted before the Closesource closes a file
      //   First argument is the LFN of the file which is being closed.
      //   Second argument is false if fallback is used; true otherwise.
      typedef signalslot::Signal<void(std::string const&, bool)> PreCloseFile;
      PreCloseFile preCloseFileSignal_;
      void watchPreCloseFile(PreCloseFile::slot_type const& iSlot) {
        preCloseFileSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPreCloseFile)

      /// signal is emitted after the source opens a file
      typedef signalslot::Signal<void(std::string const&, bool)> PostCloseFile;
      PostCloseFile postCloseFileSignal_;
      void watchPostCloseFile(PostCloseFile::slot_type const& iSlot) {
         postCloseFileSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPostCloseFile)

      typedef signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> PreModuleBeginStream;
      PreModuleBeginStream preModuleBeginStreamSignal_;
      void watchPreModuleBeginStream(PreModuleBeginStream::slot_type const& iSlot) {
         preModuleBeginStreamSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPreModuleBeginStream)
        
      typedef signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> PostModuleBeginStream;
      PostModuleBeginStream postModuleBeginStreamSignal_;
      void watchPostModuleBeginStream(PostModuleBeginStream::slot_type const& iSlot) {
         postModuleBeginStreamSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPostModuleBeginStream)
        
      typedef signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> PreModuleEndStream;
      PreModuleEndStream preModuleEndStreamSignal_;
      void watchPreModuleEndStream(PreModuleEndStream::slot_type const& iSlot) {
         preModuleEndStreamSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPreModuleEndStream)
        
      typedef signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> PostModuleEndStream;
      PostModuleEndStream postModuleEndStreamSignal_;
      void watchPostModuleEndStream(PostModuleEndStream::slot_type const& iSlot) {
         postModuleEndStreamSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPostModuleEndStream)

      typedef signalslot::Signal<void(GlobalContext const&)> PreGlobalBeginRun;
      /// signal is emitted after the Run has been created by the InputSource but before any modules have seen the Run
      PreGlobalBeginRun preGlobalBeginRunSignal_;
      void watchPreGlobalBeginRun(PreGlobalBeginRun::slot_type const& iSlot) {
         preGlobalBeginRunSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPreGlobalBeginRun)

      typedef signalslot::Signal<void(GlobalContext const&)> PostGlobalBeginRun;
      PostGlobalBeginRun postGlobalBeginRunSignal_;
      void watchPostGlobalBeginRun(PostGlobalBeginRun::slot_type const& iSlot) {
         postGlobalBeginRunSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPostGlobalBeginRun)
      
      typedef signalslot::Signal<void(GlobalContext const&)> PreGlobalEndRun;
      PreGlobalEndRun preGlobalEndRunSignal_;
      void watchPreGlobalEndRun(PreGlobalEndRun::slot_type const& iSlot) {
         preGlobalEndRunSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPreGlobalEndRun)

        typedef signalslot::Signal<void(GlobalContext const&)> PostGlobalEndRun;
      PostGlobalEndRun postGlobalEndRunSignal_;
      void watchPostGlobalEndRun(PostGlobalEndRun::slot_type const& iSlot) {
         postGlobalEndRunSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPostGlobalEndRun)
      
      typedef signalslot::Signal<void(StreamContext const&)> PreStreamBeginRun;
      PreStreamBeginRun preStreamBeginRunSignal_;
      void watchPreStreamBeginRun(PreStreamBeginRun::slot_type const& iSlot) {
         preStreamBeginRunSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPreStreamBeginRun)

      typedef signalslot::Signal<void(StreamContext const&)> PostStreamBeginRun;
      PostStreamBeginRun postStreamBeginRunSignal_;
      void watchPostStreamBeginRun(PostStreamBeginRun::slot_type const& iSlot) {
         postStreamBeginRunSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPostStreamBeginRun)
      
      typedef signalslot::Signal<void(StreamContext const&)> PreStreamEndRun;
      PreStreamEndRun preStreamEndRunSignal_;
      void watchPreStreamEndRun(PreStreamEndRun::slot_type const& iSlot) {
         preStreamEndRunSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPreStreamEndRun)

      typedef signalslot::Signal<void(StreamContext const&)> PostStreamEndRun;
      PostStreamEndRun postStreamEndRunSignal_;
      void watchPostStreamEndRun(PostStreamEndRun::slot_type const& iSlot) {
         postStreamEndRunSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPostStreamEndRun)
      
      typedef signalslot::Signal<void(GlobalContext const&)> PreGlobalBeginLumi;
      PreGlobalBeginLumi preGlobalBeginLumiSignal_;
      void watchPreGlobalBeginLumi(PreGlobalBeginLumi::slot_type const& iSlot) {
         preGlobalBeginLumiSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPreGlobalBeginLumi)

      typedef signalslot::Signal<void(GlobalContext const&)> PostGlobalBeginLumi;
      PostGlobalBeginLumi postGlobalBeginLumiSignal_;
      void watchPostGlobalBeginLumi(PostGlobalBeginLumi::slot_type const& iSlot) {
         postGlobalBeginLumiSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPostGlobalBeginLumi)
      
      typedef signalslot::Signal<void(GlobalContext const&)> PreGlobalEndLumi;
      PreGlobalEndLumi preGlobalEndLumiSignal_;
      void watchPreGlobalEndLumi(PreGlobalEndLumi::slot_type const& iSlot) {
         preGlobalEndLumiSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPreGlobalEndLumi)

      typedef signalslot::Signal<void(GlobalContext const&)> PostGlobalEndLumi;
      PostGlobalEndLumi postGlobalEndLumiSignal_;
      void watchPostGlobalEndLumi(PostGlobalEndLumi::slot_type const& iSlot) {
         postGlobalEndLumiSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPostGlobalEndLumi)
      
      typedef signalslot::Signal<void(StreamContext const&)> PreStreamBeginLumi;
      PreStreamBeginLumi preStreamBeginLumiSignal_;
      void watchPreStreamBeginLumi(PreStreamBeginLumi::slot_type const& iSlot) {
         preStreamBeginLumiSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPreStreamBeginLumi)

      typedef signalslot::Signal<void(StreamContext const&)> PostStreamBeginLumi;
      PostStreamBeginLumi postStreamBeginLumiSignal_;
      void watchPostStreamBeginLumi(PostStreamBeginLumi::slot_type const& iSlot) {
         postStreamBeginLumiSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPostStreamBeginLumi)
      
      typedef signalslot::Signal<void(StreamContext const&)> PreStreamEndLumi;
      PreStreamEndLumi preStreamEndLumiSignal_;
      void watchPreStreamEndLumi(PreStreamEndLumi::slot_type const& iSlot) {
         preStreamEndLumiSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPreStreamEndLumi)

      typedef signalslot::Signal<void(StreamContext const&)> PostStreamEndLumi;
      PostStreamEndLumi postStreamEndLumiSignal_;
      void watchPostStreamEndLumi(PostStreamEndLumi::slot_type const& iSlot) {
         postStreamEndLumiSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPostStreamEndLumi)

      typedef signalslot::Signal<void(StreamContext const&)> PreEvent;
      /// signal is emitted after the Event has been created by the InputSource but before any modules have seen the Event
      PreEvent preEventSignal_;
      void watchPreEvent(PreEvent::slot_type const& iSlot) {
         preEventSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPreEvent)

      typedef signalslot::Signal<void(StreamContext const&)> PostEvent;
      /// signal is emitted after all modules have finished processing the Event
      PostEvent postEventSignal_;
      void watchPostEvent(PostEvent::slot_type const& iSlot) {
         postEventSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPostEvent)

      /// signal is emitted before starting to process a Path for an event
        typedef signalslot::Signal<void(StreamContext const&, PathContext const&)> PrePathEvent;
      PrePathEvent prePathEventSignal_;
      void watchPrePathEvent(PrePathEvent::slot_type const& iSlot) {
        prePathEventSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPrePathEvent)

      /// signal is emitted after all modules have finished for the Path for an event
      typedef signalslot::Signal<void(StreamContext const&, PathContext const&, HLTPathStatus const&)> PostPathEvent;
      PostPathEvent postPathEventSignal_;
      void watchPostPathEvent(PostPathEvent::slot_type const& iSlot) {
         postPathEventSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_3(watchPostPathEvent)

      // OLD DELETE THIS
      typedef signalslot::ObsoleteSignal<void(EventID const&, Timestamp const&)> PreProcessEvent;
      /// signal is emitted after the Event has been created by the InputSource but before any modules have seen the Event
      PreProcessEvent preProcessEventSignal_;
      void watchPreProcessEvent(PreProcessEvent::slot_type const& iSlot) {
         preProcessEventSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPreProcessEvent)
      
      // OLD DELETE THIS
      typedef signalslot::ObsoleteSignal<void(Event const&, EventSetup const&)> PostProcessEvent;
      /// signal is emitted after all modules have finished processing the Event
      PostProcessEvent postProcessEventSignal_;
      void watchPostProcessEvent(PostProcessEvent::slot_type const& iSlot) {
         postProcessEventSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPostProcessEvent)

      // OLD DELETE THIS
      typedef signalslot::ObsoleteSignal<void(RunID const&, Timestamp const&)> PreBeginRun;
      /// signal is emitted after the Run has been created by the InputSource but before any modules have seen the Run
      PreBeginRun preBeginRunSignal_;
      void watchPreBeginRun(PreBeginRun::slot_type const& iSlot) {
         preBeginRunSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPreBeginRun)
      
      // OLD DELETE THIS
      typedef signalslot::ObsoleteSignal<void(Run const&, EventSetup const&)> PostBeginRun;
      /// signal is emitted after all modules have finished processing the beginRun 
      PostBeginRun postBeginRunSignal_;
      void watchPostBeginRun(PostBeginRun::slot_type const& iSlot) {
         postBeginRunSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPostBeginRun)

      // OLD DELETE THIS
      typedef signalslot::ObsoleteSignal<void(RunID const&, Timestamp const&)> PreEndRun;
      /// signal is emitted before the endRun is processed
      PreEndRun preEndRunSignal_;
      void watchPreEndRun(PreEndRun::slot_type const& iSlot) {
         preEndRunSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPreEndRun)
      
      // OLD DELETE THIS
      typedef signalslot::ObsoleteSignal<void(Run const&, EventSetup const&)> PostEndRun;
      /// signal is emitted after all modules have finished processing the Run
      PostEndRun postEndRunSignal_;
      void watchPostEndRun(PostEndRun::slot_type const& iSlot) {
         postEndRunSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPostEndRun)

      // OLD DELETE THIS
      typedef signalslot::ObsoleteSignal<void(LuminosityBlockID const&, Timestamp const&)> PreBeginLumi;
      /// signal is emitted after the Lumi has been created by the InputSource but before any modules have seen the Lumi
      PreBeginLumi preBeginLumiSignal_;
      void watchPreBeginLumi(PreBeginLumi::slot_type const& iSlot) {
         preBeginLumiSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPreBeginLumi)
      
      // OLD DELETE THIS
      typedef signalslot::ObsoleteSignal<void(LuminosityBlock const&, EventSetup const&)> PostBeginLumi;
      /// signal is emitted after all modules have finished processing the beginLumi
      PostBeginLumi postBeginLumiSignal_;
      void watchPostBeginLumi(PostBeginLumi::slot_type const& iSlot) {
         postBeginLumiSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPostBeginLumi)

      // OLD DELETE THIS
      typedef signalslot::ObsoleteSignal<void(LuminosityBlockID const&, Timestamp const&)> PreEndLumi;
      /// signal is emitted before the endLumi is processed
      PreEndLumi preEndLumiSignal_;
      void watchPreEndLumi(PreEndLumi::slot_type const& iSlot) {
         preEndLumiSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPreEndLumi)
      
      // OLD DELETE THIS
      typedef signalslot::ObsoleteSignal<void(LuminosityBlock const&, EventSetup const&)> PostEndLumi;
      /// signal is emitted after all modules have finished processing the Lumi
      PostEndLumi postEndLumiSignal_;
      void watchPostEndLumi(PostEndLumi::slot_type const& iSlot) {
         postEndLumiSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPostEndLumi)

      // OLD DELETE THIS
      /// signal is emitted before starting to process a Path for an event
      typedef signalslot::ObsoleteSignal<void(std::string const&)> PreProcessPath;
      PreProcessPath preProcessPathSignal_;
      void watchPreProcessPath(PreProcessPath::slot_type const& iSlot) {
        preProcessPathSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPreProcessPath)
        
      // OLD DELETE THIS
      /// signal is emitted after all modules have finished for the Path for an event
      typedef signalslot::ObsoleteSignal<void(std::string const&, HLTPathStatus const&)> PostProcessPath;
      PostProcessPath postProcessPathSignal_;
      void watchPostProcessPath(PostProcessPath::slot_type const& iSlot) {
         postProcessPathSignal_.connect_front(iSlot);
      }  
      AR_WATCH_USING_METHOD_2(watchPostProcessPath)
        
      // OLD DELETE THIS
      /// signal is emitted before starting to process a Path for beginRun
      typedef signalslot::ObsoleteSignal<void(std::string const&)> PrePathBeginRun;
      PrePathBeginRun prePathBeginRunSignal_;
      void watchPrePathBeginRun(PrePathBeginRun::slot_type const& iSlot) {
        prePathBeginRunSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPrePathBeginRun)
        
      // OLD DELETE THIS
      /// signal is emitted after all modules have finished for the Path for beginRun
      typedef signalslot::ObsoleteSignal<void(std::string const&, HLTPathStatus const&)> PostPathBeginRun;
      PostPathBeginRun postPathBeginRunSignal_;
      void watchPostPathBeginRun(PostPathBeginRun::slot_type const& iSlot) {
         postPathBeginRunSignal_.connect_front(iSlot);
      }  
      AR_WATCH_USING_METHOD_2(watchPostPathBeginRun)
        
      // OLD DELETE THIS
      /// signal is emitted before starting to process a Path for endRun
      typedef signalslot::ObsoleteSignal<void(std::string const&)> PrePathEndRun;
      PrePathEndRun prePathEndRunSignal_;
      void watchPrePathEndRun(PrePathEndRun::slot_type const& iSlot) {
        prePathEndRunSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPrePathEndRun)
        
      // OLD DELETE THIS
      /// signal is emitted after all modules have finished for the Path for endRun
      typedef signalslot::ObsoleteSignal<void(std::string const&, HLTPathStatus const&)> PostPathEndRun;
      PostPathEndRun postPathEndRunSignal_;
      void watchPostPathEndRun(PostPathEndRun::slot_type const& iSlot) {
         postPathEndRunSignal_.connect_front(iSlot);
      }  
      AR_WATCH_USING_METHOD_2(watchPostPathEndRun)
        
      // OLD DELETE THIS
      /// signal is emitted before starting to process a Path for beginLumi
      typedef signalslot::ObsoleteSignal<void(std::string const&)> PrePathBeginLumi;
      PrePathBeginLumi prePathBeginLumiSignal_;
      void watchPrePathBeginLumi(PrePathBeginLumi::slot_type const& iSlot) {
        prePathBeginLumiSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPrePathBeginLumi)
        
      // OLD DELETE THIS
      /// signal is emitted after all modules have finished for the Path for beginLumi
      typedef signalslot::ObsoleteSignal<void(std::string const&, HLTPathStatus const&)> PostPathBeginLumi;
      PostPathBeginLumi postPathBeginLumiSignal_;
      void watchPostPathBeginLumi(PostPathBeginLumi::slot_type const& iSlot) {
         postPathBeginLumiSignal_.connect_front(iSlot);
      }  
      AR_WATCH_USING_METHOD_2(watchPostPathBeginLumi)
        
      // OLD DELETE THIS
      /// signal is emitted before starting to process a Path for endRun
      typedef signalslot::ObsoleteSignal<void(std::string const&)> PrePathEndLumi;
      PrePathEndLumi prePathEndLumiSignal_;
      void watchPrePathEndLumi(PrePathEndLumi::slot_type const& iSlot) {
        prePathEndLumiSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPrePathEndLumi)
        
      // OLD DELETE THIS
      /// signal is emitted after all modules have finished for the Path for endRun
      typedef signalslot::ObsoleteSignal<void(std::string const&, HLTPathStatus const&)> PostPathEndLumi;
      PostPathEndLumi postPathEndLumiSignal_;
      void watchPostPathEndLumi(PostPathEndLumi::slot_type const& iSlot) {
         postPathEndLumiSignal_.connect_front(iSlot);
      }  
      AR_WATCH_USING_METHOD_2(watchPostPathEndLumi)

	/* Note M:
	   Concerning use of address of module descriptor
	   during functions called before/after module or source construction:
	       Unlike the case in the Run, Lumi, and Event loops,
	       the Module descriptor (often passed by pointer or reference
	       as an argument named desc) in the construction phase is NOT
	       at some permanent fixed address during the construction phase.  
	       Therefore, any optimization of caching the module name keying 
	       off of address of the descriptor will NOT be valid during 
               such functions.  mf / cj 9/11/09
	*/
	
      /// signal is emitted before the module is constructed
      typedef signalslot::Signal<void(ModuleDescription const&)> PreModuleConstruction;
      PreModuleConstruction preModuleConstructionSignal_;
      void watchPreModuleConstruction(PreModuleConstruction::slot_type const& iSlot) {
         preModuleConstructionSignal_.connect(iSlot);
      }
      // WARNING - ModuleDescription is not in fixed place.  See note M above.
      AR_WATCH_USING_METHOD_1(watchPreModuleConstruction)
         
      /// signal is emitted after the module was construction
      typedef signalslot::Signal<void(ModuleDescription const&)> PostModuleConstruction;
      PostModuleConstruction postModuleConstructionSignal_;
      void watchPostModuleConstruction(PostModuleConstruction::slot_type const& iSlot) {
         postModuleConstructionSignal_.connect_front(iSlot);
      }
      // WARNING - ModuleDescription is not in fixed place.  See note M above.
      AR_WATCH_USING_METHOD_1(watchPostModuleConstruction)

      /// signal is emitted before the module does beginJob
      typedef signalslot::Signal<void(ModuleDescription const&)> PreModuleBeginJob;
      PreModuleBeginJob preModuleBeginJobSignal_;
      void watchPreModuleBeginJob(PreModuleBeginJob::slot_type const& iSlot) {
        preModuleBeginJobSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPreModuleBeginJob)
        
      /// signal is emitted after the module had done beginJob
      typedef signalslot::Signal<void(ModuleDescription const&)> PostModuleBeginJob;
      PostModuleBeginJob postModuleBeginJobSignal_;
      void watchPostModuleBeginJob(PostModuleBeginJob::slot_type const& iSlot) {
         postModuleBeginJobSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPostModuleBeginJob)
        
      /// signal is emitted before the module does endJob
      typedef signalslot::Signal<void(ModuleDescription const&)> PreModuleEndJob;
      PreModuleEndJob preModuleEndJobSignal_;
      void watchPreModuleEndJob(PreModuleEndJob::slot_type const& iSlot) {
        preModuleEndJobSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPreModuleEndJob)
        
      /// signal is emitted after the module had done endJob
      typedef signalslot::Signal<void(ModuleDescription const&)> PostModuleEndJob;
      PostModuleEndJob postModuleEndJobSignal_;
      void watchPostModuleEndJob(PostModuleEndJob::slot_type const& iSlot) {
         postModuleEndJobSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPostModuleEndJob)

      /// signal is emitted before the module starts processing the Event
      typedef signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> PreModuleEvent;
      PreModuleEvent preModuleEventSignal_;
      void watchPreModuleEvent(PreModuleEvent::slot_type const& iSlot) {
         preModuleEventSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPreModuleEvent)
         
      /// signal is emitted after the module finished processing the Event
      typedef signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> PostModuleEvent;
      PostModuleEvent postModuleEventSignal_;
      void watchPostModuleEvent(PostModuleEvent::slot_type const& iSlot) {
         postModuleEventSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPostModuleEvent)

     /// signal is emitted after the module starts processing the Event and before a delayed get has started
     typedef signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> PreModuleEventDelayedGet;
     PreModuleEventDelayedGet preModuleEventDelayedGetSignal_;
     void watchPreModuleEventDelayedGet(PreModuleEventDelayedGet::slot_type const& iSlot) {
       preModuleEventDelayedGetSignal_.connect(iSlot);
     }
     AR_WATCH_USING_METHOD_2(watchPreModuleEventDelayedGet)
     
     /// signal is emitted after the module starts processing the Event and after a delayed get has finished
     typedef signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> PostModuleEventDelayedGet;
     PostModuleEventDelayedGet postModuleEventDelayedGetSignal_;
     void watchPostModuleEventDelayedGet(PostModuleEventDelayedGet::slot_type const& iSlot) {
       postModuleEventDelayedGetSignal_.connect_front(iSlot);
     }
     AR_WATCH_USING_METHOD_2(watchPostModuleEventDelayedGet)
     
      typedef signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> PreModuleStreamBeginRun;
      PreModuleStreamBeginRun preModuleStreamBeginRunSignal_;
      void watchPreModuleStreamBeginRun(PreModuleStreamBeginRun::slot_type const& iSlot) {
         preModuleStreamBeginRunSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPreModuleStreamBeginRun)
         
      typedef signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> PostModuleStreamBeginRun;
      PostModuleStreamBeginRun postModuleStreamBeginRunSignal_;
      void watchPostModuleStreamBeginRun(PostModuleStreamBeginRun::slot_type const& iSlot) {
         postModuleStreamBeginRunSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPostModuleStreamBeginRun)

      typedef signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> PreModuleStreamEndRun;
      PreModuleStreamEndRun preModuleStreamEndRunSignal_;
      void watchPreModuleStreamEndRun(PreModuleStreamEndRun::slot_type const& iSlot) {
         preModuleStreamEndRunSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPreModuleStreamEndRun)
         
      typedef signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> PostModuleStreamEndRun;
      PostModuleStreamEndRun postModuleStreamEndRunSignal_;
      void watchPostModuleStreamEndRun(PostModuleStreamEndRun::slot_type const& iSlot) {
         postModuleStreamEndRunSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPostModuleStreamEndRun)

      typedef signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> PreModuleStreamBeginLumi;
      PreModuleStreamBeginLumi preModuleStreamBeginLumiSignal_;
      void watchPreModuleStreamBeginLumi(PreModuleStreamBeginLumi::slot_type const& iSlot) {
         preModuleStreamBeginLumiSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPreModuleStreamBeginLumi)
         
      typedef signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> PostModuleStreamBeginLumi;
      PostModuleStreamBeginLumi postModuleStreamBeginLumiSignal_;
      void watchPostModuleStreamBeginLumi(PostModuleStreamBeginLumi::slot_type const& iSlot) {
         postModuleStreamBeginLumiSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPostModuleStreamBeginLumi)

      typedef signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> PreModuleStreamEndLumi;
      PreModuleStreamEndLumi preModuleStreamEndLumiSignal_;
      void watchPreModuleStreamEndLumi(PreModuleStreamEndLumi::slot_type const& iSlot) {
         preModuleStreamEndLumiSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPreModuleStreamEndLumi)
         
      typedef signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> PostModuleStreamEndLumi;
      PostModuleStreamEndLumi postModuleStreamEndLumiSignal_;
      void watchPostModuleStreamEndLumi(PostModuleStreamEndLumi::slot_type const& iSlot) {
         postModuleStreamEndLumiSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPostModuleStreamEndLumi)

      typedef signalslot::Signal<void(GlobalContext const&, ModuleCallingContext const&)> PreModuleGlobalBeginRun;
      PreModuleGlobalBeginRun preModuleGlobalBeginRunSignal_;
      void watchPreModuleGlobalBeginRun(PreModuleGlobalBeginRun::slot_type const& iSlot) {
         preModuleGlobalBeginRunSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPreModuleGlobalBeginRun)
         
      typedef signalslot::Signal<void(GlobalContext const&, ModuleCallingContext const&)> PostModuleGlobalBeginRun;
      PostModuleGlobalBeginRun postModuleGlobalBeginRunSignal_;
      void watchPostModuleGlobalBeginRun(PostModuleGlobalBeginRun::slot_type const& iSlot) {
         postModuleGlobalBeginRunSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPostModuleGlobalBeginRun)

      typedef signalslot::Signal<void(GlobalContext const&, ModuleCallingContext const&)> PreModuleGlobalEndRun;
      PreModuleGlobalEndRun preModuleGlobalEndRunSignal_;
      void watchPreModuleGlobalEndRun(PreModuleGlobalEndRun::slot_type const& iSlot) {
         preModuleGlobalEndRunSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPreModuleGlobalEndRun)
         
      typedef signalslot::Signal<void(GlobalContext const&, ModuleCallingContext const&)> PostModuleGlobalEndRun;
      PostModuleGlobalEndRun postModuleGlobalEndRunSignal_;
      void watchPostModuleGlobalEndRun(PostModuleGlobalEndRun::slot_type const& iSlot) {
         postModuleGlobalEndRunSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPostModuleGlobalEndRun)

      typedef signalslot::Signal<void(GlobalContext const&, ModuleCallingContext const&)> PreModuleGlobalBeginLumi;
      PreModuleGlobalBeginLumi preModuleGlobalBeginLumiSignal_;
      void watchPreModuleGlobalBeginLumi(PreModuleGlobalBeginLumi::slot_type const& iSlot) {
         preModuleGlobalBeginLumiSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPreModuleGlobalBeginLumi)
         
      typedef signalslot::Signal<void(GlobalContext const&, ModuleCallingContext const&)> PostModuleGlobalBeginLumi;
      PostModuleGlobalBeginLumi postModuleGlobalBeginLumiSignal_;
      void watchPostModuleGlobalBeginLumi(PostModuleGlobalBeginLumi::slot_type const& iSlot) {
         postModuleGlobalBeginLumiSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPostModuleGlobalBeginLumi)

      typedef signalslot::Signal<void(GlobalContext const&, ModuleCallingContext const&)> PreModuleGlobalEndLumi;
      PreModuleGlobalEndLumi preModuleGlobalEndLumiSignal_;
      void watchPreModuleGlobalEndLumi(PreModuleGlobalEndLumi::slot_type const& iSlot) {
         preModuleGlobalEndLumiSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPreModuleGlobalEndLumi)
         
      typedef signalslot::Signal<void(GlobalContext const&, ModuleCallingContext const&)> PostModuleGlobalEndLumi;
      PostModuleGlobalEndLumi postModuleGlobalEndLumiSignal_;
      void watchPostModuleGlobalEndLumi(PostModuleGlobalEndLumi::slot_type const& iSlot) {
         postModuleGlobalEndLumiSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPostModuleGlobalEndLumi)

      // OLD DELETE THIS
      /// signal is emitted before the module starts processing the Event
      typedef signalslot::ObsoleteSignal<void(ModuleDescription const&)> PreModule;
      PreModule preModuleSignal_;
      void watchPreModule(PreModule::slot_type const& iSlot) {
         preModuleSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPreModule)
         
      // OLD DELETE THIS
      /// signal is emitted after the module finished processing the Event
      typedef signalslot::ObsoleteSignal<void(ModuleDescription const&)> PostModule;
      PostModule postModuleSignal_;
      void watchPostModule(PostModule::slot_type const& iSlot) {
         postModuleSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPostModule)
         
      // OLD DELETE THIS
      /// signal is emitted before the module starts processing beginRun
      typedef signalslot::ObsoleteSignal<void(ModuleDescription const&)> PreModuleBeginRun;
      PreModuleBeginRun preModuleBeginRunSignal_;
      void watchPreModuleBeginRun(PreModuleBeginRun::slot_type const& iSlot) {
         preModuleBeginRunSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPreModuleBeginRun)
         
      // OLD DELETE THIS
      /// signal is emitted after the module finished processing beginRun
      typedef signalslot::ObsoleteSignal<void(ModuleDescription const&)> PostModuleBeginRun;
      PostModuleBeginRun postModuleBeginRunSignal_;
      void watchPostModuleBeginRun(PostModuleBeginRun::slot_type const& iSlot) {
         postModuleBeginRunSignal_.connect_front(iSlot);
         
      }
      AR_WATCH_USING_METHOD_1(watchPostModuleBeginRun)
         
      // OLD DELETE THIS
      /// signal is emitted before the module starts processing endRun
      typedef signalslot::ObsoleteSignal<void(ModuleDescription const&)> PreModuleEndRun;
      PreModuleEndRun preModuleEndRunSignal_;
      void watchPreModuleEndRun(PreModuleEndRun::slot_type const& iSlot) {
         preModuleEndRunSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPreModuleEndRun)
         
      // OLD DELETE THIS
      /// signal is emitted after the module finished processing endRun
      typedef signalslot::ObsoleteSignal<void(ModuleDescription const&)> PostModuleEndRun;
      PostModuleEndRun postModuleEndRunSignal_;
      void watchPostModuleEndRun(PostModuleEndRun::slot_type const& iSlot) {
         postModuleEndRunSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPostModuleEndRun)
         
      // OLD DELETE THIS
      /// signal is emitted before the module starts processing beginLumi
      typedef signalslot::ObsoleteSignal<void(ModuleDescription const&)> PreModuleBeginLumi;
      PreModuleBeginLumi preModuleBeginLumiSignal_;
      void watchPreModuleBeginLumi(PreModuleBeginLumi::slot_type const& iSlot) {
         preModuleBeginLumiSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPreModuleBeginLumi)
         
      // OLD DELETE THIS
      /// signal is emitted after the module finished processing beginLumi
      typedef signalslot::ObsoleteSignal<void(ModuleDescription const&)> PostModuleBeginLumi;
      PostModuleBeginLumi postModuleBeginLumiSignal_;
      void watchPostModuleBeginLumi(PostModuleBeginLumi::slot_type const& iSlot) {
         postModuleBeginLumiSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPostModuleBeginLumi)
         
      // OLD DELETE THIS
      /// signal is emitted before the module starts processing endLumi
      typedef signalslot::ObsoleteSignal<void(ModuleDescription const&)> PreModuleEndLumi;
      PreModuleEndLumi preModuleEndLumiSignal_;
      void watchPreModuleEndLumi(PreModuleEndLumi::slot_type const& iSlot) {
         preModuleEndLumiSignal_.connect(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPreModuleEndLumi)
         
      // OLD DELETE THIS
      /// signal is emitted after the module finished processing endLumi
      typedef signalslot::ObsoleteSignal<void(ModuleDescription const&)> PostModuleEndLumi;
      PostModuleEndLumi postModuleEndLumiSignal_;
      void watchPostModuleEndLumi(PostModuleEndLumi::slot_type const& iSlot) {
         postModuleEndLumiSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_1(watchPostModuleEndLumi)

      /// signal is emitted before the source is constructed
      typedef signalslot::Signal<void(ModuleDescription const&)> PreSourceConstruction;
      PreSourceConstruction preSourceConstructionSignal_;
      void watchPreSourceConstruction(PreSourceConstruction::slot_type const& iSlot) {
        preSourceConstructionSignal_.connect(iSlot);
      }
      // WARNING - ModuleDescription is not in fixed place.  See note M above.
      AR_WATCH_USING_METHOD_1(watchPreSourceConstruction)
        
      /// signal is emitted after the source was construction
      typedef signalslot::Signal<void(ModuleDescription const&)> PostSourceConstruction;
      PostSourceConstruction postSourceConstructionSignal_;
      void watchPostSourceConstruction(PostSourceConstruction::slot_type const& iSlot) {
         postSourceConstructionSignal_.connect_front(iSlot);
      }
      // WARNING - ModuleDescription is not in fixed place.  See note M above.
      AR_WATCH_USING_METHOD_1(watchPostSourceConstruction)

      /// signal is emitted before we fork the processes
      typedef signalslot::Signal<void()> PreForkReleaseResources;
      PreForkReleaseResources preForkReleaseResourcesSignal_;
      void watchPreForkReleaseResources(PreForkReleaseResources::slot_type const& iSlot) {
         preForkReleaseResourcesSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_0(watchPreForkReleaseResources)
      
      /// signal is emitted after we forked the processes
      typedef signalslot::Signal<void(unsigned int, unsigned int)> PostForkReacquireResources;
      PostForkReacquireResources postForkReacquireResourcesSignal_;
      void watchPostForkReacquireResources(PostForkReacquireResources::slot_type const& iSlot) {
         postForkReacquireResourcesSignal_.connect_front(iSlot);
      }
      AR_WATCH_USING_METHOD_2(watchPostForkReacquireResources)
      
      // ---------- member functions ---------------------------
      
      ///forwards our signals to slots connected to iOther
      void connect(ActivityRegistry& iOther);

      ///forwards our subprocess independent signals to slots connected to iOther
      ///forwards iOther's subprocess dependent signals to slots connected to this
      void connectToSubProcess(ActivityRegistry& iOther);
      
      ///copy the slots from iOther and connect them directly to our own
      /// this allows us to 'forward' signals more efficiently,
      /// BUT if iOther gains new slots after this call, we will not see them
      /// This is also careful to keep the order of the slots proper
      /// for services.
      void copySlotsFrom(ActivityRegistry& iOther);
      
   private:
      // forwards subprocess independent signals to slots connected to iOther
      void connectGlobals(ActivityRegistry& iOther);

      // forwards subprocess dependent signals to slots connected to iOther
      void connectLocals(ActivityRegistry& iOther);
   };
}
#undef AR_WATCH_USING_METHOD
#endif
