/**
   \file
   Implementation of class WorkerRegistry

   \author Stefano ARGIRO
   \version $Id: WorkerRegistry.cc,v 1.12 2006/09/01 18:16:42 wmtan Exp $
   \date 18 May 2005
*/

static const char CVSId[] = "$Id: WorkerRegistry.cc,v 1.12 2006/09/01 18:16:42 wmtan Exp $";


#include "DataFormats/Common/interface/PassID.h"
#include "DataFormats/Common/interface/ReleaseVersion.h"

#include "FWCore/Framework/src/WorkerRegistry.h"
#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/src/Factory.h"

#include <iostream>
#include <sstream>

using namespace std;
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

  string workerid =
    mangleWorkerParameters(*p.pset_, p.processName_,
			   p.releaseVersion_,p.passID_);

  WorkerMap::iterator workerIt = m_workerMap.find(workerid);
  
  // if the woker is not there, make it
  if (workerIt == m_workerMap.end()){
    
    std::auto_ptr<Worker> workerPtr=
      Factory::get()->makeWorker(p,act_reg_->preModuleConstructionSignal_,
                                 act_reg_->postModuleConstructionSignal_);
    
    workerPtr->connect(act_reg_->preModuleSignal_,act_reg_->postModuleSignal_);

    // Transfer ownership of worker to the registry 
    m_workerMap[workerid] = boost::shared_ptr<Worker>(workerPtr.release());
    return m_workerMap[workerid].get();
    
  } 
  
  return  (workerIt->second.get());

}


string WorkerRegistry::mangleWorkerParameters(ParameterSet const& parameterSet,
					      std::string const& processName,
					      ReleaseVersion const& releaseVersion,
					      PassID const& passID) {

  stringstream mangled_parameters;
  mangled_parameters<< parameterSet.toString()
                    << processName
                    << releaseVersion
                    << passID;

  return mangled_parameters.str();

}



