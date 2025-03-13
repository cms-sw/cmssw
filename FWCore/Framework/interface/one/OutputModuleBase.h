#ifndef FWCore_Framework_one_OutputModuleBase_h
#define FWCore_Framework_one_OutputModuleBase_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     OutputModuleBase
//
/**\class OutputModuleBase OutputModuleBase.h "FWCore/Framework/interface/one/OutputModuleBase.h"

 Description: Base class for all 'one' OutputModules

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Wed, 31 Jul 2013 15:37:16 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/OutputModuleCore.h"
#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"

// forward declarations
namespace edm {

  namespace one {

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
      virtual bool wantsProcessBlocks() const noexcept = 0;
      virtual bool wantsInputProcessBlocks() const noexcept = 0;
      virtual bool wantsGlobalRuns() const noexcept = 0;
      virtual bool wantsGlobalLuminosityBlocks() const noexcept = 0;
      bool wantsStreamRuns() const noexcept { return false; }
      bool wantsStreamLuminosityBlocks() const noexcept { return false; };

      virtual SerialTaskQueue* globalRunsQueue() { return nullptr; }
      virtual SerialTaskQueue* globalLuminosityBlocksQueue() { return nullptr; }
      SharedResourcesAcquirer& sharedResourcesAcquirer() { return resourcesAcquirer_; }

    protected:
      void doPreallocate(PreallocationConfiguration const&);

      void doBeginJob();
      bool doEvent(EventTransitionInfo const&, ActivityRegistry*, ModuleCallingContext const*);

      void configure(OutputModuleDescription const& desc);

    private:
      SharedResourcesAcquirer resourcesAcquirer_;
      SerialTaskQueue runQueue_;
      SerialTaskQueue luminosityBlockQueue_;

      virtual SharedResourcesAcquirer createAcquirer();

      std::string workerType() const { return "WorkerT<edm::one::OutputModuleBase>"; }

      virtual void preActionBeforeRunEventAsync(WaitingTaskHolder iTask,
                                                ModuleCallingContext const& iModuleCallingContext,
                                                Principal const& iPrincipal) const noexcept {}

      bool hasAcquire() const noexcept { return false; }
    };
  }  // namespace one
}  // namespace edm
#endif
