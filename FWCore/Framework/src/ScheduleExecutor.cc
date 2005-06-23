/**
   \file
   Implementation of ScheduleExecutor

   \author Stefano ARGIRO
   \version $Id: ScheduleExecutor.cc,v 1.1 2005/05/29 02:29:54 wmtan Exp $
   \date 19 May 2005
*/

static const char CVSId[] = "$Id: ScheduleExecutor.cc,v 1.1 2005/05/29 02:29:54 wmtan Exp $";


#include "FWCore/CoreFramework/interface/ScheduleExecutor.h"
#include "FWCore/CoreFramework/interface/EventPrincipal.h"
#include "FWCore/CoreFramework/src/Worker.h"
#include "FWCore/CoreFramework/interface/EventSetup.h"

using namespace edm;


ScheduleExecutor::ScheduleExecutor(const ScheduleExecutor::PathList& pathlist):
  m_pathlist(pathlist){}


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

      // need error handling
      if (!(*workerIt)->doWork(eventPrincipal, eventSetup)) return 1;
      
    } // for
  }// for path

  return 0;
}


