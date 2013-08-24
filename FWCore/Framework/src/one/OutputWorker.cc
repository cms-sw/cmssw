
/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/one/OutputModuleBase.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/src/one/OutputWorker.h"
#include "FWCore/Framework/src/OutputModuleCommunicator.h"
#include "FWCore/Framework/src/OutputModuleCommunicatorT.h"
#include "FWCore/Framework/src/WorkerParams.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/Utilities/interface/LuminosityBlockIndex.h"

namespace edm {

  class ProcessContext;

  namespace one {
    OutputWorker::OutputWorker(OutputModuleBase* mod,
                               ModuleDescription const& md,
                               ExceptionToActionTable const* actions):
    WorkerT<OutputModuleBase>(mod, md, actions)
    {
    }
    
    OutputWorker::~OutputWorker() {
    }
    
    std::unique_ptr<OutputModuleCommunicator>
    OutputWorker::createOutputModuleCommunicator() {
      return std::move(std::unique_ptr<OutputModuleCommunicator>{new OutputModuleCommunicatorT<edm::one::OutputModuleBase>{& this->module()}});
    }
  }
}
