/**
   \file
   Implementation of class WorkerRegistry

   \author Stefano ARGIRO
   \version $Id: WorkerRegistry.cc,v 1.8 2006/01/29 23:33:58 jbk Exp $
   \date 18 May 2005
*/

static const char CVSId[] = "$Id: WorkerRegistry.cc,v 1.8 2006/01/29 23:33:58 jbk Exp $";


#include "FWCore/Framework/src/WorkerRegistry.h"
#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/src/Factory.h"

#include <iostream>
#include <sstream>

using namespace std;
using namespace edm;

WorkerRegistry::WorkerRegistry():
  act_reg_(new ActivityRegistry)
{
}

WorkerRegistry::WorkerRegistry(boost::shared_ptr<ActivityRegistry> areg):
  act_reg_(areg)
{
}

WorkerRegistry:: ~WorkerRegistry(){

  for (WorkerMap::iterator workerIt = m_workerMap.begin();
	 workerIt != m_workerMap.end() ; ++ workerIt)
    delete workerIt->second;
}


Worker* WorkerRegistry::getWorker(const WorkerParams& p) {

  string workerid= 
    mangleWorkerParameters(*p.pset_, p.processName_,
			   p.versionNumber__,p.pass_);

  WorkerMap::iterator workerIt = m_workerMap.find(workerid);
  
  // if the woker is not there, make it
  if (workerIt == m_workerMap.end()){
    
    std::auto_ptr<Worker> workerPtr=
      Factory::get()->makeWorker(p,act_reg_->preModuleConstructionSignal_,
                                 act_reg_->preModuleConstructionSignal_);
    
    workerPtr->connect(act_reg_->preModuleSignal_,act_reg_->postModuleSignal_);

    Worker* w =  workerPtr.get(); // take ownership
    m_workerMap[workerid] = w;
    workerPtr.release();
    return w;
    
  } 
  
  return  (workerIt->second);

}


string WorkerRegistry::mangleWorkerParameters(ParameterSet const& parameterSet,
					      std::string const& processName,
					      unsigned long versionNumber,
					      unsigned long pass) {

  stringstream mangled_parameters;
  mangled_parameters<< parameterSet.toString()
                    << processName
                    << versionNumber
                    << pass;

  return mangled_parameters.str();

}



