#ifndef FWCore_Framework_ProducerWorker_h
#define FWCore_Framework_ProducerWorker_h

/*----------------------------------------------------------------------
  
ProducerWorker: The EDProducer as the schedule sees it.  The job of
this object is to call the producer, collect up the results, and
feed them into the event.
According to our current definition, a single producer can only
appear in one worker.

$Id: ProducerWorker.h,v 1.23 2008/01/11 20:30:09 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>

#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/src/WorkerT.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edm {

  class ProducerWorker : public WorkerT<EDProducer> {
  public:
    ProducerWorker(std::auto_ptr<EDProducer>,
		   ModuleDescription const&,
		   WorkerParams const&);

    virtual ~ProducerWorker();

  private:
    virtual bool implDoWork(EventPrincipal& e, EventSetup const& c,
			    BranchActionType,
			    CurrentProcessingContext const* cpc);
    virtual bool implDoWork(RunPrincipal& rp, EventSetup const& c,
			    BranchActionType bat,
			    CurrentProcessingContext const* cpc);
    virtual bool implDoWork(LuminosityBlockPrincipal& lbp, EventSetup const& c,
			    BranchActionType bat,
			    CurrentProcessingContext const* cpc);

    virtual std::string workerType() const;
  };

  template <> 
  struct WorkerType<EDProducer> {
    typedef EDProducer ModuleType;
    typedef ProducerWorker worker_type;
  };

}

#endif
