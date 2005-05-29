/**
   \file
   Implementation of class ScheduleBuilder

   \author Stefano ARGIRO
   \version $Id: ScheduleBuilder.cc,v 1.5 2005/05/26 08:28:53 argiro Exp $
   \date 18 May 2005
*/

static const char CVSId[] = "$Id: ScheduleBuilder.cc,v 1.5 2005/05/26 08:28:53 argiro Exp $";


#include "FWCore/CoreFramework/interface/ScheduleBuilder.h"
#include "FWCore/CoreFramework/src/WorkerRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <iostream>

using namespace edm;
using namespace std;

ScheduleBuilder::ScheduleBuilder(ParameterSet const& processDesc): 
  m_processDesc(processDesc){

  // fill PathList
  // use the limited-scope function  makeProcessPSet

  const vector<string>& modulenames = 
    m_processDesc.getVString("temporary_single_path");

  const std::string& processName =  m_processDesc.getString("process_name");

  WorkerList workerList;

  for (vector<string>::const_iterator nameIt=modulenames.begin();
       nameIt!=modulenames.end(); 
       ++nameIt){

    ParameterSet const& module_pset= m_processDesc.getPSet(*nameIt);
    
#warning version and pass are hardcoded
    unsigned long version = 1;
    unsigned long pass    = 1;
    
    Worker* worker= 
      WorkerRegistry::get()->getWorker(module_pset,processName,version,pass);
    
    workerList.push_back(worker);

  }// for

  m_PathList.push_back(workerList);

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
