#ifndef FWCore_Framework_one_OutputWorker_h
#define FWCore_Framework_one_OutputWorker_h

/*----------------------------------------------------------------------
  
OutputWorker: The one::OutputModuleBase as the schedule sees it.  The job of
this object is to call the output module.

According to our current definition, a single output module can only
appear in one worker.
----------------------------------------------------------------------*/

#include <memory>

#include "FWCore/Framework/interface/one/OutputModuleBase.h"
#include "FWCore/Framework/src/WorkerT.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

namespace edm {
  namespace one {

  class OutputWorker : public WorkerT<OutputModuleBase> {
  public:
    OutputWorker(OutputModuleBase* mod,
		 ModuleDescription const&,
		 ExceptionToActionTable const* actions);

    virtual ~OutputWorker();

    std::unique_ptr<OutputModuleCommunicator> createOutputModuleCommunicator() override;
    
  };
  }

}

#endif
