/**
   \file
   Implementation of class ScheduleBuilder

   \author Stefano ARGIRO
   \version $Id: ScheduleBuilder.cc,v 1.19.2.2 2006/07/05 23:57:18 wmtan Exp $
   \date 18 May 2005
*/

static const char CVSId[] = "$Id: ScheduleBuilder.cc,v 1.19.2.2 2006/07/05 23:57:18 wmtan Exp $";


#include "FWCore/Framework/interface/ScheduleBuilder.h"
#include "FWCore/Framework/src/WorkerRegistry.h"
#include "DataFormats/Common/interface/ProductRegistry.h"
#include "FWCore/Framework/src/WorkerParams.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/GetPassID.h"
#include "FWCore/Utilities/interface/GetReleaseVersion.h"
#include "PluginManager/PluginManager.h"

#include <cstdlib>
#include <iostream>

using namespace edm;
using namespace std;

ScheduleBuilder::ScheduleBuilder(ParameterSet const& processDesc,
				 WorkerRegistry& wregistry,
				 ProductRegistry& pregistry,
				 ActionTable& actions): 
  m_processDesc(processDesc) {


  seal::PluginManager::get()->initialise();

  // fill PathList

  // get the list of available paths from the processDesc
  const vector<string>& pathnames = 
    m_processDesc.getParameter<vector<string> >("@paths");

  // loop on paths
  for (vector<string>::const_iterator pathIt = pathnames.begin(); 
       pathIt != pathnames.end(); ++pathIt){
  
    const vector<string>& modulenames = 
      m_processDesc.getParameter<vector<string> >(*pathIt);
    
    const std::string& processName = m_processDesc.getParameter<string>("@process_name");
    
    WorkerList workerList;
    // loop on workers
    for (vector<string>::const_iterator nameIt=modulenames.begin();
	 nameIt!=modulenames.end(); 
	 ++nameIt){
           
      try{
        m_processDesc.getParameter<ParameterSet>(*nameIt);
      }catch(const cms::Exception&) {
        throw edm::Exception(errors::Configuration,"PathError:")<<"no module with label '"<<*nameIt<<"' is defined in configuration file, but is used in path '"
        <<*pathIt<<"'.\n Please check spelling.";
      }
      ParameterSet const& module_pset= m_processDesc.getParameter<ParameterSet>(*nameIt);

      WorkerParams params(m_processDesc, module_pset, pregistry, actions,
			  processName, getReleaseVersion(), getPassID());
    
      Worker* worker= wregistry.getWorker(params);
      
      workerList.push_back(worker);
      
    }// for
    
    m_pathList.push_back(workerList);
    
  } // loop on paths
    
  pregistry.setProductIDs();
}


const ScheduleBuilder::PathList& ScheduleBuilder::getPathList() const{
 
  return m_pathList;
}



// Configure (x)emacs for this file ...
// Local Variables:
// mode:c++
// compile-command: "make -C .. -k"
// End:
