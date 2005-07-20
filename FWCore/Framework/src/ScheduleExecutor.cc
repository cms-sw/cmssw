/**
   \file
   Implementation of ScheduleExecutor

   \author Stefano ARGIRO
   \version $Id: ScheduleExecutor.cc,v 1.3 2005/07/14 22:50:53 wmtan Exp $
   \date 19 May 2005
*/

static const char CVSId[] = "$Id: ScheduleExecutor.cc,v 1.3 2005/07/14 22:50:53 wmtan Exp $";


#include "FWCore/Framework/interface/ScheduleExecutor.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Actions.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <iostream>

using namespace edm;
using namespace std;


ScheduleExecutor::ScheduleExecutor(const ScheduleExecutor::PathList& pathlist,
				   const ActionTable& tab):
  m_pathlist(pathlist),
  m_actions(&tab)
{}


int 
ScheduleExecutor::runOneEvent(EventPrincipal& eventPrincipal, 
			      EventSetup const& eventSetup){


  // loop on paths
  for (PathList::const_iterator pathIt = m_pathlist.begin();
       pathIt != m_pathlist.end(); 
       ++pathIt){
    
    // loop on workers
    for (WorkerList::const_iterator workerIt = pathIt->begin();
	 workerIt != pathIt->end(); 
	 ++workerIt){

      try
	{
	  // stop this path if worker returns false
	  if (!(*workerIt)->doWork(eventPrincipal, eventSetup)) break;
	}
      catch(cms::Exception& e)
	{
	  actions::ActionCodes code = m_actions->find(e.rootCause());

	  if(code==actions::IgnoreCompletely)
	    {
	      // change to error logger - print the path name FIXME
	      cerr << "Ignoring exception from event " << eventPrincipal.id()
		   << " in path XXX"
		   << "\nmessage:\n" << e.what()
		   << endl;
	      continue;
	    }
	  else if(code==actions::FailPath)
	    {
	      // mark path as failed FIXME
	      cerr << "Exception caused path end from event "
		   << eventPrincipal.id()
		   << " in path XXX"
		   << "\nmessage:\n" << e.what()
		   << endl;
	      break;
	    }
	  else
	    throw edm::Exception(errors::ScheduleExecutionFailure,
				 "ProcessingStopped",e);
	  
	}
      
    } // for
  }// for path

  return 0;
}


