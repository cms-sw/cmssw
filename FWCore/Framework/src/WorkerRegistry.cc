/**
   \file
   Implementation of class WorkerRegistry

   \author Stefano ARGIRO
   \version $Id: WorkerRegistry.cc,v 1.21 2009/02/13 02:43:59 wmtan Exp $
   \date 18 May 2005
*/

static const char CVSId[] = "$Id: WorkerRegistry.cc,v 1.21 2009/02/13 02:43:59 wmtan Exp $";

#include "FWCore/Framework/src/WorkerRegistry.h"
#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/src/Factory.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

namespace edm {

  WorkerRegistry::WorkerRegistry(boost::shared_ptr<ActivityRegistry> areg) :
    m_workerMap(),
    actReg_(areg)
  {
  }

  WorkerRegistry:: ~WorkerRegistry() {
    m_workerMap.clear();
  }

  void WorkerRegistry::clear() {
    m_workerMap.clear();
  }

  Worker* WorkerRegistry::getWorker(WorkerParams const& p, std::string const& moduleLabel) {

    WorkerMap::iterator workerIt = m_workerMap.find(moduleLabel);
  
    // if the worker is not there, make it
    if (workerIt == m_workerMap.end()){
    
      std::auto_ptr<Worker> workerPtr=
        Factory::get()->makeWorker(p,actReg_->preModuleConstructionSignal_,
                                   actReg_->postModuleConstructionSignal_);
    
      workerPtr->setActivityRegistry(actReg_);

      // Transfer ownership of worker to the registry 
      m_workerMap[moduleLabel].reset(workerPtr.release());
      return m_workerMap[moduleLabel].get(); 
    } 
    return (workerIt->second.get());
  }
}
