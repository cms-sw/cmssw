/**
   \file
   Implementation of ScheduleExecutor

   \author Stefano ARGIRO
   \version $Id: ScheduleExecutor.cc,v 1.2 2005/06/23 19:59:48 wmtan Exp $
   \date 19 May 2005
*/

static const char CVSId[] = "$Id: ScheduleExecutor.cc,v 1.2 2005/06/23 19:59:48 wmtan Exp $";


#include "FWCore/Framework/interface/ScheduleExecutor.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/interface/EventSetup.h"

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


