/**
   \file
   Implementation of class WorkerRegistry

   \author Stefano ARGIRO
   \version $Id: WorkerRegistry.cc,v 1.17 2007/06/29 03:43:22 wmtan Exp $
   \date 18 May 2005
*/

static const char CVSId[] = "$Id: WorkerRegistry.cc,v 1.17 2007/06/29 03:43:22 wmtan Exp $";


#include "FWCore/Framework/src/WorkerRegistry.h"
#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/src/Factory.h"

#include <sstream>

using namespace edm;

WorkerRegistry::WorkerRegistry() :
  m_workerMap(),
  act_reg_(new ActivityRegistry)
{
}

WorkerRegistry::WorkerRegistry(boost::shared_ptr<ActivityRegistry> areg) :
  m_workerMap(),
  act_reg_(areg)
{
}

WorkerRegistry:: ~WorkerRegistry() {
  m_workerMap.clear();
}

void WorkerRegistry::clear() {
  m_workerMap.clear();
}

Worker* WorkerRegistry::getWorker(const WorkerParams& p) {

  std::string workerid =
    mangleWorkerParameters(*p.pset_, p.processName_,
			   p.releaseVersion_,p.passID_);

  WorkerMap::iterator workerIt = m_workerMap.find(workerid);
  
  // if the woker is not there, make it
  if (workerIt == m_workerMap.end()){
    
    std::auto_ptr<Worker> workerPtr=
      Factory::get()->makeWorker(p,act_reg_->preModuleConstructionSignal_,
                                 act_reg_->postModuleConstructionSignal_);
    
    workerPtr->connect(act_reg_->preModuleSignal_,act_reg_->postModuleSignal_,
                       act_reg_->preModuleBeginJobSignal_,act_reg_->postModuleBeginJobSignal_,
                       act_reg_->preModuleEndJobSignal_,act_reg_->postModuleEndJobSignal_);

    // Transfer ownership of worker to the registry 
    m_workerMap[workerid].reset(workerPtr.release());
    return m_workerMap[workerid].get();
    
  } 
  
  return  (workerIt->second.get());

}


std::string WorkerRegistry::mangleWorkerParameters(ParameterSet const& parameterSet,
					      std::string const& processName,
					      ReleaseVersion const& releaseVersion,
					      PassID const& passID) {

  std::stringstream mangled_parameters;
  mangled_parameters<< parameterSet.toString()
                    << processName
                    << releaseVersion
                    << passID;

  return mangled_parameters.str();

}



