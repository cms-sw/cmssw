/**
   \file
   Implementation of class ScheduleBuilder

   \author Stefano ARGIRO
   \version $Id: ScheduleBuilder.cc,v 1.5 2005/06/14 00:00:49 wmtan Exp $
   \date 18 May 2005
*/

static const char CVSId[] = "$Id: ScheduleBuilder.cc,v 1.5 2005/06/14 00:00:49 wmtan Exp $";


#include "FWCore/CoreFramework/interface/ScheduleBuilder.h"
#include "FWCore/CoreFramework/src/WorkerRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PluginManager/PluginManager.h"

#include <iostream>

using namespace edm;
using namespace std;

ScheduleBuilder::ScheduleBuilder(ParameterSet const& processDesc): 
  m_processDesc(processDesc){


  seal::PluginManager::get()->initialise();

  // fill PathList

  // get the list of available paths from the processDesc
  const vector<string>& pathnames = 
    getParameter<vector<string> >(m_processDesc, "paths");

  // loop on paths
  for (vector<string>::const_iterator pathIt = pathnames.begin(); 
       pathIt != pathnames.end(); ++pathIt){
  
    const vector<string>& modulenames = 
      getParameter<vector<string> >(m_processDesc, *pathIt);
    
    const std::string& processName = getParameter<string>(m_processDesc, "process_name");
    
    WorkerList workerList;
    // loop on workers
    for (vector<string>::const_iterator nameIt=modulenames.begin();
	 nameIt!=modulenames.end(); 
	 ++nameIt){
           
      ParameterSet const& module_pset= getParameter<ParameterSet>(m_processDesc, *nameIt);
   
#warning version and pass are hardcoded
      unsigned long version = 1;
      unsigned long pass    = 1;
    
      Worker* worker= 
	WorkerRegistry::get()->getWorker(module_pset,processName,version,pass);
      
      workerList.push_back(worker);
      
    }// for
    
    m_PathList.push_back(workerList);
    
  } // loop on paths
  
  validate();
  
}


const ScheduleBuilder::PathList& ScheduleBuilder::getPathList() const{
 
  return m_PathList;
}


bool  ScheduleBuilder::validate(){return true;}


// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "make -C .. -k"
// End:
