#ifndef FWCore_Framework_global_OutputModuleBase_h
#define FWCore_Framework_global_OutputModuleBase_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     OutputModuleBase
//
/**\class OutputModuleBase OutputModuleBase.h "FWCore/Framework/interface/global/OutputModuleBase.h"

 Description: Base class for all 'global' OutputModules

 Usage:
    <usage>

*/
//
//

// system include files

// user include files
#include "FWCore/Framework/interface/OutputModuleCore.h"

// forward declarations
namespace edm {

  namespace global {

    class OutputModuleBase : public core::OutputModuleCore {
    public:
      template <typename U>
      friend class edm::maker::ModuleHolderT;
      template <typename T>
      friend class ::edm::WorkerT;
      template <typename T>
      friend class ::edm::OutputModuleCommunicatorT;
      typedef OutputModuleBase ModuleType;

      explicit OutputModuleBase(ParameterSet const& pset);

      OutputModuleBase(OutputModuleBase const&) = delete;             // Disallow copying and moving
      OutputModuleBase& operator=(OutputModuleBase const&) = delete;  // Disallow copying and moving

      //Output modules always need writeRun and writeLumi to be called
      bool wantsGlobalRuns() const { return true; }
      bool wantsGlobalLuminosityBlocks() const { return true; }

      virtual bool wantsProcessBlocks() const = 0;
      virtual bool wantsInputProcessBlocks() const = 0;
      virtual bool wantsStreamRuns() const = 0;
      virtual bool wantsStreamLuminosityBlocks() const = 0;

    protected:
      void doPreallocate(PreallocationConfiguration const&);

      void doBeginJob();

      void doBeginStream(StreamID id) { doBeginStream_(id); }
      void doEndStream(StreamID id) { doEndStream_(id); }

      bool doEvent(EventTransitionInfo const&, ActivityRegistry*, ModuleCallingContext const*);
      void doAcquire(EventTransitionInfo const&,
                     ActivityRegistry*,
                     ModuleCallingContext const*,
                     WaitingTaskWithArenaHolder&);
      //For now this is a placeholder
      /*virtual*/ void preActionBeforeRunEventAsync(WaitingTaskHolder iTask,
                                                    ModuleCallingContext const& iModuleCallingContext,
                                                    Principal const& iPrincipal) const {}

    private:
      std::string workerType() const { return "WorkerT<edm::global::OutputModuleBase>"; }

      virtual void preallocStreams(unsigned int) {}
      virtual void preallocate(PreallocationConfiguration const&) {}
      virtual void doBeginStream_(StreamID) {}
      virtual void doEndStream_(StreamID) {}
      virtual void doStreamBeginRun_(StreamID, RunForOutput const&, EventSetup const&) {}
      virtual void doStreamEndRun_(StreamID, RunForOutput const&, EventSetup const&) {}
      virtual void doStreamEndRunSummary_(StreamID, RunForOutput const&, EventSetup const&) {}
      virtual void doStreamBeginLuminosityBlock_(StreamID, LuminosityBlockForOutput const&, EventSetup const&) {}
      virtual void doStreamEndLuminosityBlock_(StreamID, LuminosityBlockForOutput const&, EventSetup const&) {}
      virtual void doStreamEndLuminosityBlockSummary_(StreamID, LuminosityBlockForOutput const&, EventSetup const&) {}

      virtual void doBeginRunSummary_(RunForOutput const&, EventSetup const&) {}
      virtual void doEndRunSummary_(RunForOutput const&, EventSetup const&) {}
      virtual void doBeginLuminosityBlockSummary_(LuminosityBlockForOutput const&, EventSetup const&) {}
      virtual void doEndLuminosityBlockSummary_(LuminosityBlockForOutput const&, EventSetup const&) {}
      virtual void doAcquire_(StreamID, EventForOutput const&, WaitingTaskWithArenaHolder&) {}

      virtual bool hasAcquire() const { return false; }
    };
  }  // namespace global
}  // namespace edm
#endif
