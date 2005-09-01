#ifndef Framework_ProducerWorker_h
#define Framework_ProducerWorker_h

/*----------------------------------------------------------------------
  
ProducerWorker: The EDProducer as the schedule sees it.  The job of
this object is to call the producer, collect up the results, and
feed them into the event.
According to our current definition, a single producer can only
appear in one worker.

$Id: ProducerWorker.h,v 1.10 2005/09/01 04:30:52 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>

#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/src/WorkerParams.h"
#include "FWCore/Framework/interface/ModuleDescription.h"

namespace edm
{
  class ActionTable;
  class ParameterSet;

  class ProducerWorker : public Worker
  {
  public:
    ProducerWorker(std::auto_ptr<EDProducer>,
		   const ModuleDescription&,
		   const WorkerParams&);

    virtual ~ProducerWorker();

    template <class ModType>
    static std::auto_ptr<EDProducer> makeOne(const ModuleDescription& md,
					     const WorkerParams& wp);
  private:
    virtual bool doWork(EventPrincipal& e, EventSetup const& c);

    virtual void beginJob(EventSetup const&) ;
    virtual void endJob() ;
    
    ModuleDescription md_;
    boost::shared_ptr<EDProducer> producer_;
    const ActionTable* actions_; // memory assumed to be managed elsewhere
  };

  template <> 
  struct WorkerType<EDProducer>
  {
    typedef EDProducer ModuleType;
    typedef ProducerWorker worker_type;
  };

  template <class ModType>
  std::auto_ptr<EDProducer> ProducerWorker::makeOne(const ModuleDescription&,
						const WorkerParams& wp)
  {
    return std::auto_ptr<EDProducer>(new ModType(*wp.pset_));
  }

}

#endif // Framework_ProducerWorker_h
