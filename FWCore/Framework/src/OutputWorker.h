#ifndef Framework_OutputWorker_h
#define Framework_OutputWorker_h

/*----------------------------------------------------------------------
  
OutputWorker: The OutputModule as the schedule sees it.  The job of
this object is to call the output module.

According to our current definition, a single output module can only
appear in one worker.
----------------------------------------------------------------------*/

#include <memory>

#include "FWCore/Framework/interface/OutputModule.h"
#include "FWCore/Framework/src/WorkerT.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

namespace edm {

  class OutputWorker : public WorkerT<OutputModule> {
  public:
    OutputWorker(std::unique_ptr<OutputModule>&& mod,
		 ModuleDescription const&,
		 WorkerParams const&);

    virtual ~OutputWorker();

    std::unique_ptr<OutputModuleCommunicator> createOutputModuleCommunicator() override;
    
  };

}

#endif
