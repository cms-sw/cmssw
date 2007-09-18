#ifndef FWCore_Framework_ProducerWorker_h
#define FWCore_Framework_ProducerWorker_h

/*----------------------------------------------------------------------
  
ProducerWorker: The EDProducer as the schedule sees it.  The job of
this object is to call the producer, collect up the results, and
feed them into the event.
According to our current definition, a single producer can only
appear in one worker.

$Id: ProducerWorker.h,v 1.21 2007/06/08 23:52:59 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>

#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/src/Worker.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edm {

  class ProducerWorker : public Worker {
  public:
    ProducerWorker(std::auto_ptr<EDProducer>,
		   ModuleDescription const&,
		   WorkerParams const&);

    virtual ~ProducerWorker();

    template <class ModType>
    static std::auto_ptr<EDProducer> makeOne(ModuleDescription const& md,
					     WorkerParams const& wp);
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

    virtual void implBeginJob(EventSetup const&) ;
    virtual void implEndJob() ;
    virtual std::string workerType() const;
    
    boost::shared_ptr<EDProducer> producer_;
  };

  template <> 
  struct WorkerType<EDProducer> {
    typedef EDProducer ModuleType;
    typedef ProducerWorker worker_type;
  };

  template <class ModType>
  std::auto_ptr<EDProducer> ProducerWorker::makeOne(ModuleDescription const& md,
						WorkerParams const& wp) {
    ParameterSetDescription desc;
    ModType::fillDescription(desc);
    desc.validate(*wp.pset_);
    std::auto_ptr<ModType> producer = std::auto_ptr<ModType>(new ModType(*wp.pset_));
    producer->setModuleDescription(md);
    return std::auto_ptr<EDProducer>(producer.release());
  }

}

#endif
