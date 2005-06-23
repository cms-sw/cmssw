/**
   \file
   Implementation of class WorkerRegistry

   \author Stefano ARGIRO
   \version $Id: WorkerRegistry.cc,v 1.1 2005/05/29 02:29:54 wmtan Exp $
   \date 18 May 2005
*/

static const char CVSId[] = "$Id: WorkerRegistry.cc,v 1.1 2005/05/29 02:29:54 wmtan Exp $";


#include "FWCore/CoreFramework/src/WorkerRegistry.h"
#include "FWCore/CoreFramework/src/Worker.h"
#include "FWCore/CoreFramework/src/Factory.h"

#include <iostream>
#include <sstream>

using namespace std;
using namespace edm;

WorkerRegistry* WorkerRegistry::m_singleInstance=0;

WorkerRegistry* WorkerRegistry::get(){

  if (!m_singleInstance){
    m_singleInstance= new WorkerRegistry;
  }
  return m_singleInstance;

}


WorkerRegistry:: ~WorkerRegistry(){

  for (WorkerMap::iterator workerIt = m_workerMap.begin();
	 workerIt != m_workerMap.end() ; ++ workerIt)
    delete workerIt->second;

}


Worker* WorkerRegistry::getWorker(ParameterSet const& parameterSet,
				  std::string const& processName,
				  unsigned long versionNumber,
				  unsigned long pass){

  string workerid= 
    mangleWorkerParameters(parameterSet, processName,versionNumber,pass);

  WorkerMap::iterator workerIt = m_workerMap.find(workerid);
  
  // if the woker is not there, make it
  if (workerIt == m_workerMap.end()){
    
    std::auto_ptr<Worker> workerPtr=
      Factory::get()->makeWorker(parameterSet,processName,versionNumber,pass);
    
    Worker* w =  workerPtr.release(); // take ownership
    m_workerMap[workerid] = w;
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



