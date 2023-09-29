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

There are four little details you should remember when adding new signals
to this file that go beyond the obvious cut and paste type of edits.
  1. The number at the end of the AR_WATCH_USING_METHOD_X macro definition
  is the number of function arguments. It will not compile if you use the
  wrong number there.
  2. Use connect or connect_front depending on whether the callback function
  should be called for different services in the order the Services were
  constructed or in reverse order. Begin signals are usually forward and
  End signals in reverse, but if the service does not depend on other services
  and vice versa this does not matter.
  3. The signal needs to be added to either connectGlobals or connectLocals
  in the ActivityRegistry.cc file, depending on whether a signal is seen
  by children or parents when there are SubProcesses. For example, source
  signals are only generated in the top level process and should be seen
  by all child SubProcesses so they are in connectGlobals. Most signals
  however belong in connectLocals. It does not really matter in jobs
  without at least one SubProcess.
  4. Each signal also needs to be added in copySlotsFrom in
  ActivityRegistry.cc. Whether it uses copySlotsToFrom or
  copySlotsToFromReverse depends on the same ordering issue as the connect
  or connect_front choice in item 2 above.
*/
//
// Original Author:  Chris Jones
//         Created:  Mon Sep  5 19:53:09 EDT 2005
//

// system include files
#include <functional>
#include <string>

// user include files
#include "FWCore/ServiceRegistry/interface/TerminationOrigin.h"
#include "FWCore/Utilities/interface/LuminosityBlockIndex.h"
#include "FWCore/Utilities/interface/RunIndex.h"
#include "FWCore/Utilities/interface/Signal.h"
#include "FWCore/Utilities/interface/StreamID.h"

#define AR_WATCH_USING_METHOD_0(method)               \
  template <class TClass, class TMethod>              \
  void method(TClass* iObject, TMethod iMethod) {     \
    method(std::bind(std::mem_fn(iMethod), iObject)); \
  }
#define AR_WATCH_USING_METHOD_1(method)                                      \
  template <class TClass, class TMethod>                                     \
  void method(TClass* iObject, TMethod iMethod) {                            \
    method(std::bind(std::mem_fn(iMethod), iObject, std::placeholders::_1)); \
  }
#define AR_WATCH_USING_METHOD_2(method)                                                             \
  template <class TClass, class TMethod>                                                            \
  void method(TClass* iObject, TMethod iMethod) {                                                   \
    method(std::bind(std::mem_fn(iMethod), iObject, std::placeholders::_1, std::placeholders::_2)); \
  }
#define AR_WATCH_USING_METHOD_3(method)                                                                       \
  template <class TClass, class TMethod>                                                                      \
  void method(TClass* iObject, TMethod iMethod) {                                                             \
    method(std::bind(                                                                                         \
        std::mem_fn(iMethod), iObject, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3)); \
  }
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
  class IOVSyncValue;
  class HLTPathStatus;
  class GlobalContext;
  class StreamContext;
  class PathContext;
  class ProcessContext;
  class ModuleCallingContext;
  class PathsAndConsumesOfModulesBase;
  class ESModuleCallingContext;
  namespace eventsetup {
    struct ComponentDescription;
    class DataKey;
    class EventSetupRecordKey;
  }  // namespace eventsetup
  namespace service {
    class SystemBounds;
  }

  namespace signalslot {
    void throwObsoleteSignalException();

    template <class T>
    class ObsoleteSignal {
    public:
      typedef std::function<T> slot_type;

      ObsoleteSignal() = default;

      template <typename U>
      void connect(U /*  iFunc */) {
        throwObsoleteSignalException();
      }

      template <typename U>
      void connect_front(U /*  iFunc*/) {
        throwObsoleteSignalException();
      }
    };
  }  // namespace signalslot
  class ActivityRegistry {
  public:
    ActivityRegistry() {}
    ActivityRegistry(ActivityRegistry const&) = delete;             // Disallow copying and moving
    ActivityRegistry& operator=(ActivityRegistry const&) = delete;  // Disallow copying and moving

    // ---------- signals ------------------------------------
    typedef signalslot::Signal<void(service::SystemBounds const&)> Preallocate;
    ///signal is emitted before beginJob
    Preallocate preallocateSignal_;
    void watchPreallocate(Preallocate::slot_type const& iSlot) { preallocateSignal_.connect(iSlot); }
    AR_WATCH_USING_METHOD_1(watchPreallocate)

    typedef signalslot::Signal<void(PathsAndConsumesOfModulesBase const&, ProcessContext const&)> PreBeginJob;
    ///signal is emitted before all modules have gotten their beginJob called
    PreBeginJob preBeginJobSignal_;
    ///convenience function for attaching to signal
    void watchPreBeginJob(PreBeginJob::slot_type const& iSlot) { preBeginJobSignal_.connect(iSlot); }
    AR_WATCH_USING_METHOD_2(watchPreBeginJob)

    typedef signalslot::Signal<void()> PostBeginJob;
    ///signal is emitted after all modules have gotten their beginJob called
    PostBeginJob postBeginJobSignal_;
    ///convenience function for attaching to signal
    void watchPostBeginJob(PostBeginJob::slot_type const& iSlot) { postBeginJobSignal_.connect(iSlot); }
    AR_WATCH_USING_METHOD_0(watchPostBeginJob)

    typedef signalslot::Signal<void()> PreEndJob;
    ///signal is emitted before any modules have gotten their endJob called
    PreEndJob preEndJobSignal_;
    void watchPreEndJob(PreEndJob::slot_type const& iSlot) { preEndJobSignal_.connect_front(iSlot); }
    AR_WATCH_USING_METHOD_0(watchPreEndJob)

    typedef signalslot::Signal<void()> PostEndJob;
    ///signal is emitted after all modules have gotten their endJob called
    PostEndJob postEndJobSignal_;
    void watchPostEndJob(PostEndJob::slot_type const& iSlot) { postEndJobSignal_.connect_front(iSlot); }
    AR_WATCH_USING_METHOD_0(watchPostEndJob)

    typedef signalslot::Signal<void()> JobFailure;
    /// signal is emitted if event processing or end-of-job
    /// processing fails with an uncaught exception.
    JobFailure jobFailureSignal_;
    ///convenience function for attaching to signal
    void watchJobFailure(JobFailure::slot_type const& iSlot) { jobFailureSignal_.connect_front(iSlot); }
    AR_WATCH_USING_METHOD_0(watchJobFailure)

    /// signal is emitted before the source starts creating an Event
    typedef signalslot::Signal<void(StreamID)> PreSourceEvent;
    PreSourceEvent preSourceSignal_;
    void watchPreSourceEvent(PreSourceEvent::slot_type const& iSlot) { preSourceSignal_.connect(iSlot); }
    AR_WATCH_USING_METHOD_1(watchPreSourceEvent)

    /// signal is emitted after the source starts creating an Event
    typedef signalslot::Signal<void(StreamID)> PostSourceEvent;
    PostSourceEvent postSourceSignal_;
    void watchPostSourceEvent(PostSourceEvent::slot_type const& iSlot) { postSourceSignal_.connect_front(iSlot); }
    AR_WATCH_USING_METHOD_1(watchPostSourceEvent)

    /// signal is emitted before the source starts creating a Lumi
    typedef signalslot::Signal<void(LuminosityBlockIndex)> PreSourceLumi;
    PreSourceLumi preSourceLumiSignal_;
    void watchPreSourceLumi(PreSourceLumi::slot_type const& iSlot) { preSourceLumiSignal_.connect(iSlot); }
    AR_WATCH_USING_METHOD_1(watchPreSourceLumi)

    /// signal is emitted after the source starts creating a Lumi
    typedef signalslot::Signal<void(LuminosityBlockIndex)> PostSourceLumi;
    PostSourceLumi postSourceLumiSignal_;
    void watchPostSourceLumi(PostSourceLumi::slot_type const& iSlot) { postSourceLumiSignal_.connect_front(iSlot); }
    AR_WATCH_USING_METHOD_1(watchPostSourceLumi)

    /// signal is emitted before the source starts creating a Run
    typedef signalslot::Signal<void(RunIndex)> PreSourceRun;
    PreSourceRun preSourceRunSignal_;
    void watchPreSourceRun(PreSourceRun::slot_type const& iSlot) { preSourceRunSignal_.connect(iSlot); }
    AR_WATCH_USING_METHOD_1(watchPreSourceRun)

    /// signal is emitted after the source starts creating a Run
    typedef signalslot::Signal<void(RunIndex)> PostSourceRun;
    PostSourceRun postSourceRunSignal_;
    void watchPostSourceRun(PostSourceRun::slot_type const& iSlot) { postSourceRunSignal_.connect_front(iSlot); }
    AR_WATCH_USING_METHOD_1(watchPostSourceRun)

    /// signal is emitted before the source starts creating a ProcessBlock
    typedef signalslot::Signal<void()> PreSourceProcessBlock;
    PreSourceProcessBlock preSourceProcessBlockSignal_;
    void watchPreSourceProcessBlock(PreSourceProcessBlock::slot_type const& iSlot) {
      preSourceProcessBlockSignal_.connect(iSlot);
    }
    AR_WATCH_USING_METHOD_0(watchPreSourceProcessBlock)

    /// signal is emitted after the source starts creating a ProcessBlock
    typedef signalslot::Signal<void(std::string const&)> PostSourceProcessBlock;
    PostSourceProcessBlock postSourceProcessBlockSignal_;
    void watchPostSourceProcessBlock(PostSourceProcessBlock::slot_type const& iSlot) {
      postSourceProcessBlockSignal_.connect_front(iSlot);
    }
    AR_WATCH_USING_METHOD_1(watchPostSourceProcessBlock)

    /// signal is emitted before the source opens a file
    typedef signalslot::Signal<void(std::string const&)> PreOpenFile;
    PreOpenFile preOpenFileSignal_;
    void watchPreOpenFile(PreOpenFile::slot_type const& iSlot) { preOpenFileSignal_.connect(iSlot); }
    AR_WATCH_USING_METHOD_1(watchPreOpenFile)

    /// signal is emitted after the source opens a file
    //   Note this is only done for a primary file, not a secondary one.
    typedef signalslot::Signal<void(std::string const&)> PostOpenFile;
    PostOpenFile postOpenFileSignal_;
    void watchPostOpenFile(PostOpenFile::slot_type const& iSlot) { postOpenFileSignal_.connect_front(iSlot); }
    AR_WATCH_USING_METHOD_1(watchPostOpenFile)

    /// signal is emitted before the source closes a file
    //   First argument is the LFN of the file which is being closed.
    typedef signalslot::Signal<void(std::string const&)> PreCloseFile;
    PreCloseFile preCloseFileSignal_;
    void watchPreCloseFile(PreCloseFile::slot_type const& iSlot) { preCloseFileSignal_.connect(iSlot); }
    AR_WATCH_USING_METHOD_1(watchPreCloseFile)

    /// signal is emitted after the source closes a file
    typedef signalslot::Signal<void(std::string const&)> PostCloseFile;
    PostCloseFile postCloseFileSignal_;
    void watchPostCloseFile(PostCloseFile::slot_type const& iSlot) { postCloseFileSignal_.connect_front(iSlot); }
    AR_WATCH_USING_METHOD_1(watchPostCloseFile)

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

    typedef signalslot::Signal<void(GlobalContext const&)> PreBeginProcessBlock;
    PreBeginProcessBlock preBeginProcessBlockSignal_;
    void watchPreBeginProcessBlock(PreBeginProcessBlock::slot_type const& iSlot) {
      preBeginProcessBlockSignal_.connect(iSlot);
    }
    AR_WATCH_USING_METHOD_1(watchPreBeginProcessBlock)

    typedef signalslot::Signal<void(GlobalContext const&)> PostBeginProcessBlock;
    PostBeginProcessBlock postBeginProcessBlockSignal_;
    void watchPostBeginProcessBlock(PostBeginProcessBlock::slot_type const& iSlot) {
      postBeginProcessBlockSignal_.connect_front(iSlot);
    }
    AR_WATCH_USING_METHOD_1(watchPostBeginProcessBlock)

    typedef signalslot::Signal<void(GlobalContext const&)> PreAccessInputProcessBlock;
    PreAccessInputProcessBlock preAccessInputProcessBlockSignal_;
    void watchPreAccessInputProcessBlock(PreAccessInputProcessBlock::slot_type const& iSlot) {
      preAccessInputProcessBlockSignal_.connect(iSlot);
    }
    AR_WATCH_USING_METHOD_1(watchPreAccessInputProcessBlock)

    typedef signalslot::Signal<void(GlobalContext const&)> PostAccessInputProcessBlock;
    PostAccessInputProcessBlock postAccessInputProcessBlockSignal_;
    void watchPostAccessInputProcessBlock(PostAccessInputProcessBlock::slot_type const& iSlot) {
      postAccessInputProcessBlockSignal_.connect_front(iSlot);
    }
    AR_WATCH_USING_METHOD_1(watchPostAccessInputProcessBlock)

    typedef signalslot::Signal<void(GlobalContext const&)> PreEndProcessBlock;
    PreEndProcessBlock preEndProcessBlockSignal_;
    void watchPreEndProcessBlock(PreEndProcessBlock::slot_type const& iSlot) {
      preEndProcessBlockSignal_.connect(iSlot);
    }
    AR_WATCH_USING_METHOD_1(watchPreEndProcessBlock)

    typedef signalslot::Signal<void(GlobalContext const&)> PostEndProcessBlock;
    PostEndProcessBlock postEndProcessBlockSignal_;
    void watchPostEndProcessBlock(PostEndProcessBlock::slot_type const& iSlot) {
      postEndProcessBlockSignal_.connect_front(iSlot);
    }
    AR_WATCH_USING_METHOD_1(watchPostEndProcessBlock)

    typedef signalslot::Signal<void()> BeginProcessing;
    /// signal is emitted just before the transitions from the Source will begin to be processed
    BeginProcessing beginProcessingSignal_;
    void watchBeginProcessing(BeginProcessing::slot_type const& iSlot) { beginProcessingSignal_.connect(iSlot); }
    AR_WATCH_USING_METHOD_0(watchBeginProcessing)

    typedef signalslot::Signal<void()> EndProcessing;
    /// signal is emitted after all work has been done processing all source transitions
    EndProcessing endProcessingSignal_;
    void watchEndProcessing(EndProcessing::slot_type const& iSlot) { endProcessingSignal_.connect(iSlot); }
    AR_WATCH_USING_METHOD_0(watchEndProcessing)

    typedef signalslot::Signal<void(GlobalContext const&)> PreGlobalBeginRun;
    /// signal is emitted after the Run has been created by the InputSource but before any modules have seen the Run
    PreGlobalBeginRun preGlobalBeginRunSignal_;
    void watchPreGlobalBeginRun(PreGlobalBeginRun::slot_type const& iSlot) { preGlobalBeginRunSignal_.connect(iSlot); }
    AR_WATCH_USING_METHOD_1(watchPreGlobalBeginRun)

    typedef signalslot::Signal<void(GlobalContext const&)> PostGlobalBeginRun;
    PostGlobalBeginRun postGlobalBeginRunSignal_;
    void watchPostGlobalBeginRun(PostGlobalBeginRun::slot_type const& iSlot) {
      postGlobalBeginRunSignal_.connect_front(iSlot);
    }
    AR_WATCH_USING_METHOD_1(watchPostGlobalBeginRun)

    typedef signalslot::Signal<void(GlobalContext const&)> PreGlobalEndRun;
    PreGlobalEndRun preGlobalEndRunSignal_;
    void watchPreGlobalEndRun(PreGlobalEndRun::slot_type const& iSlot) { preGlobalEndRunSignal_.connect(iSlot); }
    AR_WATCH_USING_METHOD_1(watchPreGlobalEndRun)

    typedef signalslot::Signal<void(GlobalContext const&)> PostGlobalEndRun;
    PostGlobalEndRun postGlobalEndRunSignal_;
    void watchPostGlobalEndRun(PostGlobalEndRun::slot_type const& iSlot) {
      postGlobalEndRunSignal_.connect_front(iSlot);
    }
    AR_WATCH_USING_METHOD_1(watchPostGlobalEndRun)

    typedef signalslot::Signal<void(GlobalContext const&)> PreWriteProcessBlock;
    PreWriteProcessBlock preWriteProcessBlockSignal_;
    void watchPreWriteProcessBlock(PreWriteProcessBlock::slot_type const& iSlot) {
      preWriteProcessBlockSignal_.connect(iSlot);
    }
    AR_WATCH_USING_METHOD_1(watchPreWriteProcessBlock)

    typedef signalslot::Signal<void(GlobalContext const&)> PostWriteProcessBlock;
    PostWriteProcessBlock postWriteProcessBlockSignal_;
    void watchPostWriteProcessBlock(PostWriteProcessBlock::slot_type const& iSlot) {
      postWriteProcessBlockSignal_.connect_front(iSlot);
    }
    AR_WATCH_USING_METHOD_1(watchPostWriteProcessBlock)

    typedef signalslot::Signal<void(GlobalContext const&)> PreGlobalWriteRun;
    PreGlobalWriteRun preGlobalWriteRunSignal_;
    void watchPreGlobalWriteRun(PreGlobalWriteRun::slot_type const& iSlot) { preGlobalWriteRunSignal_.connect(iSlot); }
    AR_WATCH_USING_METHOD_1(watchPreGlobalWriteRun)

    typedef signalslot::Signal<void(GlobalContext const&)> PostGlobalWriteRun;
    PostGlobalWriteRun postGlobalWriteRunSignal_;
    void watchPostGlobalWriteRun(PostGlobalWriteRun::slot_type const& iSlot) {
      postGlobalWriteRunSignal_.connect_front(iSlot);
    }
    AR_WATCH_USING_METHOD_1(watchPostGlobalWriteRun)

    typedef signalslot::Signal<void(StreamContext const&)> PreStreamBeginRun;
    PreStreamBeginRun preStreamBeginRunSignal_;
    void watchPreStreamBeginRun(PreStreamBeginRun::slot_type const& iSlot) { preStreamBeginRunSignal_.connect(iSlot); }
    AR_WATCH_USING_METHOD_1(watchPreStreamBeginRun)

    typedef signalslot::Signal<void(StreamContext const&)> PostStreamBeginRun;
    PostStreamBeginRun postStreamBeginRunSignal_;
    void watchPostStreamBeginRun(PostStreamBeginRun::slot_type const& iSlot) {
      postStreamBeginRunSignal_.connect_front(iSlot);
    }
    AR_WATCH_USING_METHOD_1(watchPostStreamBeginRun)

    typedef signalslot::Signal<void(StreamContext const&)> PreStreamEndRun;
    PreStreamEndRun preStreamEndRunSignal_;
    void watchPreStreamEndRun(PreStreamEndRun::slot_type const& iSlot) { preStreamEndRunSignal_.connect(iSlot); }
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
    void watchPreGlobalEndLumi(PreGlobalEndLumi::slot_type const& iSlot) { preGlobalEndLumiSignal_.connect(iSlot); }
    AR_WATCH_USING_METHOD_1(watchPreGlobalEndLumi)

    typedef signalslot::Signal<void(GlobalContext const&)> PostGlobalEndLumi;
    PostGlobalEndLumi postGlobalEndLumiSignal_;
    void watchPostGlobalEndLumi(PostGlobalEndLumi::slot_type const& iSlot) {
      postGlobalEndLumiSignal_.connect_front(iSlot);
    }
    AR_WATCH_USING_METHOD_1(watchPostGlobalEndLumi)

    typedef signalslot::Signal<void(GlobalContext const&)> PreGlobalWriteLumi;
    PreGlobalEndLumi preGlobalWriteLumiSignal_;
    void watchPreGlobalWriteLumi(PreGlobalWriteLumi::slot_type const& iSlot) {
      preGlobalWriteLumiSignal_.connect(iSlot);
    }
    AR_WATCH_USING_METHOD_1(watchPreGlobalWriteLumi)

    typedef signalslot::Signal<void(GlobalContext const&)> PostGlobalWriteLumi;
    PostGlobalEndLumi postGlobalWriteLumiSignal_;
    void watchPostGlobalWriteLumi(PostGlobalEndLumi::slot_type const& iSlot) {
      postGlobalWriteLumiSignal_.connect_front(iSlot);
    }
    AR_WATCH_USING_METHOD_1(watchPostGlobalWriteLumi)

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
    void watchPreStreamEndLumi(PreStreamEndLumi::slot_type const& iSlot) { preStreamEndLumiSignal_.connect(iSlot); }
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
    void watchPreEvent(PreEvent::slot_type const& iSlot) { preEventSignal_.connect(iSlot); }
    AR_WATCH_USING_METHOD_1(watchPreEvent)

    typedef signalslot::Signal<void(StreamContext const&)> PostEvent;
    /// signal is emitted after all modules have finished processing the Event
    PostEvent postEventSignal_;
    void watchPostEvent(PostEvent::slot_type const& iSlot) { postEventSignal_.connect_front(iSlot); }
    AR_WATCH_USING_METHOD_1(watchPostEvent)

    /// signal is emitted before starting to process a Path for an event
    typedef signalslot::Signal<void(StreamContext const&, PathContext const&)> PrePathEvent;
    PrePathEvent prePathEventSignal_;
    void watchPrePathEvent(PrePathEvent::slot_type const& iSlot) { prePathEventSignal_.connect(iSlot); }
    AR_WATCH_USING_METHOD_2(watchPrePathEvent)

    /// signal is emitted after all modules have finished for the Path for an event
    typedef signalslot::Signal<void(StreamContext const&, PathContext const&, HLTPathStatus const&)> PostPathEvent;
    PostPathEvent postPathEventSignal_;
    void watchPostPathEvent(PostPathEvent::slot_type const& iSlot) { postPathEventSignal_.connect_front(iSlot); }
    AR_WATCH_USING_METHOD_3(watchPostPathEvent)

    /// signal is emitted when began processing a stream transition and
    ///  then we began terminating the application
    typedef signalslot::Signal<void(StreamContext const&, TerminationOrigin)> PreStreamEarlyTermination;
    PreStreamEarlyTermination preStreamEarlyTerminationSignal_;
    void watchPreStreamEarlyTermination(PreStreamEarlyTermination::slot_type const& iSlot) {
      preStreamEarlyTerminationSignal_.connect(iSlot);
    }
    AR_WATCH_USING_METHOD_2(watchPreStreamEarlyTermination)

    /// signal is emitted if a began processing a global transition and
    ///  then we began terminating the application
    typedef signalslot::Signal<void(GlobalContext const&, TerminationOrigin)> PreGlobalEarlyTermination;
    PreGlobalEarlyTermination preGlobalEarlyTerminationSignal_;
    void watchPreGlobalEarlyTermination(PreGlobalEarlyTermination::slot_type const& iSlot) {
      preGlobalEarlyTerminationSignal_.connect(iSlot);
    }
    AR_WATCH_USING_METHOD_2(watchPreGlobalEarlyTermination)

    /// signal is emitted if while communicating with a source we began terminating
    ///  the application
    typedef signalslot::Signal<void(TerminationOrigin)> PreSourceEarlyTermination;
    PreSourceEarlyTermination preSourceEarlyTerminationSignal_;
    void watchPreSourceEarlyTermination(PreSourceEarlyTermination::slot_type const& iSlot) {
      preSourceEarlyTerminationSignal_.connect(iSlot);
    }
    AR_WATCH_USING_METHOD_1(watchPreSourceEarlyTermination)

    /// signal is emitted after the ESModule is registered with EventSetupProvider
    using PostESModuleRegistration = signalslot::Signal<void(eventsetup::ComponentDescription const&)>;
    PostESModuleRegistration postESModuleRegistrationSignal_;
    void watchPostESModuleRegistration(PostESModuleRegistration::slot_type const& iSlot) {
      postESModuleRegistrationSignal_.connect(iSlot);
    }
    AR_WATCH_USING_METHOD_1(watchPostESModuleRegistration)

    /// signal is emitted when a new IOV may be needed so we queue a task to do that
    using ESSyncIOVQueuing = signalslot::Signal<void(IOVSyncValue const&)>;
    ESSyncIOVQueuing esSyncIOVQueuingSignal_;
    void watchESSyncIOVQueuing(ESSyncIOVQueuing::slot_type const& iSlot) { esSyncIOVQueuingSignal_.connect(iSlot); }
    AR_WATCH_USING_METHOD_1(watchESSyncIOVQueuing)

    /// signal is emitted just before a new IOV is synchronized
    using PreESSyncIOV = signalslot::Signal<void(IOVSyncValue const&)>;
    PreESSyncIOV preESSyncIOVSignal_;
    void watchPreESSyncIOV(PreESSyncIOV::slot_type const& iSlot) { preESSyncIOVSignal_.connect(iSlot); }
    AR_WATCH_USING_METHOD_1(watchPreESSyncIOV)

    /// signal is emitted just after a new IOV is synchronized
    using PostESSyncIOV = signalslot::Signal<void(IOVSyncValue const&)>;
    PostESSyncIOV postESSyncIOVSignal_;
    void watchPostESSyncIOV(PostESSyncIOV::slot_type const& iSlot) { postESSyncIOVSignal_.connect(iSlot); }
    AR_WATCH_USING_METHOD_1(watchPostESSyncIOV)

    /// signal is emitted before the esmodule starts processing and before prefetching has started
    typedef signalslot::Signal<void(eventsetup::EventSetupRecordKey const&, ESModuleCallingContext const&)>
        PreESModulePrefetching;
    PreESModulePrefetching preESModulePrefetchingSignal_;
    void watchPreESModulePrefetching(PreESModulePrefetching::slot_type const& iSlot) {
      preESModulePrefetchingSignal_.connect(iSlot);
    }
    AR_WATCH_USING_METHOD_2(watchPreESModulePrefetching)

    /// signal is emitted before the esmodule starts processing  and after prefetching has finished
    typedef signalslot::Signal<void(eventsetup::EventSetupRecordKey const&, ESModuleCallingContext const&)>
        PostESModulePrefetching;
    PostESModulePrefetching postESModulePrefetchingSignal_;
    void watchPostESModulePrefetching(PostESModulePrefetching::slot_type const& iSlot) {
      postESModulePrefetchingSignal_.connect_front(iSlot);
    }
    AR_WATCH_USING_METHOD_2(watchPostESModulePrefetching)

    /// signal is emitted before the esmodule starts processing
    typedef signalslot::Signal<void(eventsetup::EventSetupRecordKey const&, ESModuleCallingContext const&)> PreESModule;
    PreESModule preESModuleSignal_;
    void watchPreESModule(PreESModule::slot_type const& iSlot) { preESModuleSignal_.connect(iSlot); }
    AR_WATCH_USING_METHOD_2(watchPreESModule)

    /// signal is emitted after the esmodule finished processing
    typedef signalslot::Signal<void(eventsetup::EventSetupRecordKey const&, ESModuleCallingContext const&)> PostESModule;
    PostESModule postESModuleSignal_;
    void watchPostESModule(PostESModule::slot_type const& iSlot) { postESModuleSignal_.connect_front(iSlot); }
    AR_WATCH_USING_METHOD_2(watchPostESModule)

    /// signal is emitted before an esmodule starts running its acquire method
    typedef signalslot::Signal<void(eventsetup::EventSetupRecordKey const&, ESModuleCallingContext const&)>
        PreESModuleAcquire;
    PreESModuleAcquire preESModuleAcquireSignal_;
    void watchPreESModuleAcquire(PreESModuleAcquire::slot_type const& iSlot) {
      preESModuleAcquireSignal_.connect(iSlot);
    }
    AR_WATCH_USING_METHOD_2(watchPreESModuleAcquire)

    /// signal is emitted after an esmodule finishes running its acquire method
    typedef signalslot::Signal<void(eventsetup::EventSetupRecordKey const&, ESModuleCallingContext const&)>
        PostESModuleAcquire;
    PostESModuleAcquire postESModuleAcquireSignal_;
    void watchPostESModuleAcquire(PostESModuleAcquire::slot_type const& iSlot) {
      postESModuleAcquireSignal_.connect_front(iSlot);
    }
    AR_WATCH_USING_METHOD_2(watchPostESModuleAcquire)

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

    /// signal is emitted before the module is destructed, only for modules deleted before beginJob
    typedef signalslot::Signal<void(ModuleDescription const&)> PreModuleDestruction;
    PreModuleDestruction preModuleDestructionSignal_;
    void watchPreModuleDestruction(PreModuleDestruction::slot_type const& iSlot) {
      preModuleDestructionSignal_.connect(iSlot);
    }
    // note: ModuleDescription IS in the fixed place. See note M above.
    AR_WATCH_USING_METHOD_1(watchPreModuleDestruction)

    /// signal is emitted after the module is destructed, only for modules deleted before beginJob
    typedef signalslot::Signal<void(ModuleDescription const&)> PostModuleDestruction;
    PostModuleDestruction postModuleDestructionSignal_;
    void watchPostModuleDestruction(PostModuleDestruction::slot_type const& iSlot) {
      postModuleDestructionSignal_.connect_front(iSlot);
    }
    // WARNING - ModuleDescription is not in fixed place.  See note M above.
    AR_WATCH_USING_METHOD_1(watchPostModuleDestruction)

    /// signal is emitted before the module does beginJob
    typedef signalslot::Signal<void(ModuleDescription const&)> PreModuleBeginJob;
    PreModuleBeginJob preModuleBeginJobSignal_;
    void watchPreModuleBeginJob(PreModuleBeginJob::slot_type const& iSlot) { preModuleBeginJobSignal_.connect(iSlot); }
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
    void watchPreModuleEndJob(PreModuleEndJob::slot_type const& iSlot) { preModuleEndJobSignal_.connect(iSlot); }
    AR_WATCH_USING_METHOD_1(watchPreModuleEndJob)

    /// signal is emitted after the module had done endJob
    typedef signalslot::Signal<void(ModuleDescription const&)> PostModuleEndJob;
    PostModuleEndJob postModuleEndJobSignal_;
    void watchPostModuleEndJob(PostModuleEndJob::slot_type const& iSlot) {
      postModuleEndJobSignal_.connect_front(iSlot);
    }
    AR_WATCH_USING_METHOD_1(watchPostModuleEndJob)

    /// signal is emitted before the module starts processing the Event and before prefetching has started
    typedef signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> PreModuleEventPrefetching;
    PreModuleEventPrefetching preModuleEventPrefetchingSignal_;
    void watchPreModuleEventPrefetching(PreModuleEventPrefetching::slot_type const& iSlot) {
      preModuleEventPrefetchingSignal_.connect(iSlot);
    }
    AR_WATCH_USING_METHOD_2(watchPreModuleEventPrefetching)

    /// signal is emitted before the module starts processing the Event and after prefetching has finished
    typedef signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> PostModuleEventPrefetching;
    PostModuleEventPrefetching postModuleEventPrefetchingSignal_;
    void watchPostModuleEventPrefetching(PostModuleEventPrefetching::slot_type const& iSlot) {
      postModuleEventPrefetchingSignal_.connect_front(iSlot);
    }
    AR_WATCH_USING_METHOD_2(watchPostModuleEventPrefetching)

    /// signal is emitted before the module starts processing the Event
    typedef signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> PreModuleEvent;
    PreModuleEvent preModuleEventSignal_;
    void watchPreModuleEvent(PreModuleEvent::slot_type const& iSlot) { preModuleEventSignal_.connect(iSlot); }
    AR_WATCH_USING_METHOD_2(watchPreModuleEvent)

    /// signal is emitted after the module finished processing the Event
    typedef signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> PostModuleEvent;
    PostModuleEvent postModuleEventSignal_;
    void watchPostModuleEvent(PostModuleEvent::slot_type const& iSlot) { postModuleEventSignal_.connect_front(iSlot); }
    AR_WATCH_USING_METHOD_2(watchPostModuleEvent)

    /// signal is emitted before the module starts the acquire method for the Event
    typedef signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> PreModuleEventAcquire;
    PreModuleEventAcquire preModuleEventAcquireSignal_;
    void watchPreModuleEventAcquire(PreModuleEventAcquire::slot_type const& iSlot) {
      preModuleEventAcquireSignal_.connect(iSlot);
    }
    AR_WATCH_USING_METHOD_2(watchPreModuleEventAcquire)

    /// signal is emitted after the module finishes the acquire method for the Event
    typedef signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> PostModuleEventAcquire;
    PostModuleEventAcquire postModuleEventAcquireSignal_;
    void watchPostModuleEventAcquire(PostModuleEventAcquire::slot_type const& iSlot) {
      postModuleEventAcquireSignal_.connect_front(iSlot);
    }
    AR_WATCH_USING_METHOD_2(watchPostModuleEventAcquire)

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

    /// signal is emitted after the module starts processing the Event, after a delayed get has started, and before a source read
    typedef signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> PreEventReadFromSource;
    PreEventReadFromSource preEventReadFromSourceSignal_;
    void watchPreEventReadFromSource(PreEventReadFromSource::slot_type const& iSlot) {
      preEventReadFromSourceSignal_.connect(iSlot);
    }
    AR_WATCH_USING_METHOD_2(watchPreEventReadFromSource)

    /// signal is emitted after the module starts processing the Event, after a delayed get has started, and after a source read
    typedef signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> PostEventReadFromSource;
    PostEventReadFromSource postEventReadFromSourceSignal_;
    void watchPostEventReadFromSource(PostEventReadFromSource::slot_type const& iSlot) {
      postEventReadFromSourceSignal_.connect_front(iSlot);
    }
    AR_WATCH_USING_METHOD_2(watchPostEventReadFromSource)

    /// signal is emitted before the module starts processing a non-Event stream transition and before prefetching has started
    typedef signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> PreModuleStreamPrefetching;
    PreModuleStreamPrefetching preModuleStreamPrefetchingSignal_;
    void watchPreModuleStreamPrefetching(PreModuleStreamPrefetching::slot_type const& iSlot) {
      preModuleStreamPrefetchingSignal_.connect(iSlot);
    }
    AR_WATCH_USING_METHOD_2(watchPreModuleStreamPrefetching)

    /// signal is emitted before the module starts processing a non-Event stream transition and after prefetching has finished
    typedef signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> PostModuleStreamPrefetching;
    PostModuleStreamPrefetching postModuleStreamPrefetchingSignal_;
    void watchPostModuleStreamPrefetching(PostModuleStreamPrefetching::slot_type const& iSlot) {
      postModuleStreamPrefetchingSignal_.connect_front(iSlot);
    }
    AR_WATCH_USING_METHOD_2(watchPostModuleStreamPrefetching)

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

    typedef signalslot::Signal<void(GlobalContext const&, ModuleCallingContext const&)> PreModuleBeginProcessBlock;
    PreModuleBeginProcessBlock preModuleBeginProcessBlockSignal_;
    void watchPreModuleBeginProcessBlock(PreModuleBeginProcessBlock::slot_type const& iSlot) {
      preModuleBeginProcessBlockSignal_.connect(iSlot);
    }
    AR_WATCH_USING_METHOD_2(watchPreModuleBeginProcessBlock)

    typedef signalslot::Signal<void(GlobalContext const&, ModuleCallingContext const&)> PostModuleBeginProcessBlock;
    PostModuleBeginProcessBlock postModuleBeginProcessBlockSignal_;
    void watchPostModuleBeginProcessBlock(PostModuleBeginProcessBlock::slot_type const& iSlot) {
      postModuleBeginProcessBlockSignal_.connect_front(iSlot);
    }
    AR_WATCH_USING_METHOD_2(watchPostModuleBeginProcessBlock)

    typedef signalslot::Signal<void(GlobalContext const&, ModuleCallingContext const&)> PreModuleAccessInputProcessBlock;
    PreModuleAccessInputProcessBlock preModuleAccessInputProcessBlockSignal_;
    void watchPreModuleAccessInputProcessBlock(PreModuleAccessInputProcessBlock::slot_type const& iSlot) {
      preModuleAccessInputProcessBlockSignal_.connect(iSlot);
    }
    AR_WATCH_USING_METHOD_2(watchPreModuleAccessInputProcessBlock)

    typedef signalslot::Signal<void(GlobalContext const&, ModuleCallingContext const&)>
        PostModuleAccessInputProcessBlock;
    PostModuleAccessInputProcessBlock postModuleAccessInputProcessBlockSignal_;
    void watchPostModuleAccessInputProcessBlock(PostModuleAccessInputProcessBlock::slot_type const& iSlot) {
      postModuleAccessInputProcessBlockSignal_.connect_front(iSlot);
    }
    AR_WATCH_USING_METHOD_2(watchPostModuleAccessInputProcessBlock)

    typedef signalslot::Signal<void(GlobalContext const&, ModuleCallingContext const&)> PreModuleEndProcessBlock;
    PreModuleEndProcessBlock preModuleEndProcessBlockSignal_;
    void watchPreModuleEndProcessBlock(PreModuleEndProcessBlock::slot_type const& iSlot) {
      preModuleEndProcessBlockSignal_.connect(iSlot);
    }
    AR_WATCH_USING_METHOD_2(watchPreModuleEndProcessBlock)

    typedef signalslot::Signal<void(GlobalContext const&, ModuleCallingContext const&)> PostModuleEndProcessBlock;
    PostModuleEndProcessBlock postModuleEndProcessBlockSignal_;
    void watchPostModuleEndProcessBlock(PostModuleEndProcessBlock::slot_type const& iSlot) {
      postModuleEndProcessBlockSignal_.connect_front(iSlot);
    }
    AR_WATCH_USING_METHOD_2(watchPostModuleEndProcessBlock)

    /// signal is emitted before the module starts processing a global transition and before prefetching has started
    typedef signalslot::Signal<void(GlobalContext const&, ModuleCallingContext const&)> PreModuleGlobalPrefetching;
    PreModuleGlobalPrefetching preModuleGlobalPrefetchingSignal_;
    void watchPreModuleGlobalPrefetching(PreModuleGlobalPrefetching::slot_type const& iSlot) {
      preModuleGlobalPrefetchingSignal_.connect(iSlot);
    }
    AR_WATCH_USING_METHOD_2(watchPreModuleGlobalPrefetching)

    /// signal is emitted before the module starts processing a global transition and after prefetching has finished
    typedef signalslot::Signal<void(GlobalContext const&, ModuleCallingContext const&)> PostModuleGlobalPrefetching;
    PostModuleGlobalPrefetching postModuleGlobalPrefetchingSignal_;
    void watchPostModuleGlobalPrefetching(PostModuleGlobalPrefetching::slot_type const& iSlot) {
      postModuleGlobalPrefetchingSignal_.connect_front(iSlot);
    }
    AR_WATCH_USING_METHOD_2(watchPostModuleGlobalPrefetching)

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

    typedef signalslot::Signal<void(GlobalContext const&, ModuleCallingContext const&)> PreModuleWriteProcessBlock;
    PreModuleWriteProcessBlock preModuleWriteProcessBlockSignal_;
    void watchPreModuleWriteProcessBlock(PreModuleWriteProcessBlock::slot_type const& iSlot) {
      preModuleWriteProcessBlockSignal_.connect(iSlot);
    }
    AR_WATCH_USING_METHOD_2(watchPreModuleWriteProcessBlock)

    typedef signalslot::Signal<void(GlobalContext const&, ModuleCallingContext const&)> PostModuleWriteProcessBlock;
    PostModuleWriteProcessBlock postModuleWriteProcessBlockSignal_;
    void watchPostModuleWriteProcessBlock(PostModuleWriteProcessBlock::slot_type const& iSlot) {
      postModuleWriteProcessBlockSignal_.connect_front(iSlot);
    }
    AR_WATCH_USING_METHOD_2(watchPostModuleWriteProcessBlock)

    typedef signalslot::Signal<void(GlobalContext const&, ModuleCallingContext const&)> PreModuleWriteRun;
    PreModuleWriteRun preModuleWriteRunSignal_;
    void watchPreModuleWriteRun(PreModuleWriteRun::slot_type const& iSlot) { preModuleWriteRunSignal_.connect(iSlot); }
    AR_WATCH_USING_METHOD_2(watchPreModuleWriteRun)

    typedef signalslot::Signal<void(GlobalContext const&, ModuleCallingContext const&)> PostModuleWriteRun;
    PostModuleWriteRun postModuleWriteRunSignal_;
    void watchPostModuleWriteRun(PostModuleWriteRun::slot_type const& iSlot) {
      postModuleWriteRunSignal_.connect_front(iSlot);
    }
    AR_WATCH_USING_METHOD_2(watchPostModuleWriteRun)

    typedef signalslot::Signal<void(GlobalContext const&, ModuleCallingContext const&)> PreModuleWriteLumi;
    PreModuleWriteLumi preModuleWriteLumiSignal_;
    void watchPreModuleWriteLumi(PreModuleWriteLumi::slot_type const& iSlot) {
      preModuleWriteLumiSignal_.connect(iSlot);
    }
    AR_WATCH_USING_METHOD_2(watchPreModuleWriteLumi)

    typedef signalslot::Signal<void(GlobalContext const&, ModuleCallingContext const&)> PostModuleWriteLumi;
    PostModuleWriteLumi postModuleWriteLumiSignal_;
    void watchPostModuleWriteLumi(PostModuleWriteLumi::slot_type const& iSlot) {
      postModuleWriteLumiSignal_.connect_front(iSlot);
    }
    AR_WATCH_USING_METHOD_2(watchPostModuleWriteLumi)

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
}  // namespace edm
#undef AR_WATCH_USING_METHOD
#endif
