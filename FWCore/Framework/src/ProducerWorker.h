#ifndef EDM_EDPRODUCERWORKER_INCLUDED
#define EDM_EDPRODUCERWORKER_INCLUDED

/*----------------------------------------------------------------------
  
ProducerWorker: The EDProducer as the schedule sees it.  The job of
this object is to call the producer, collect up the results, and
feed them into the event.
According to our current definition, a single producer can only
appear in one worker.

$Id: ProducerWorker.h,v 1.6 2005/07/20 03:00:36 jbk Exp $

----------------------------------------------------------------------*/

#include <memory>

#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/interface/ModuleDescription.h"

namespace edm
{
  class ActionTable;
  class WorkerParams;

  class ProducerWorker : public Worker
  {
  public:
    ProducerWorker(std::auto_ptr<EDProducer>,
		   const ModuleDescription&,
		   const WorkerParams&);

    virtual ~ProducerWorker();

  private:
    virtual bool doWork(EventPrincipal& e, EventSetup const& c);

    virtual void beginJob( EventSetup const& ) ;
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

}

#endif // EDM_EDPRODUCERWORKER_INCLUDED
