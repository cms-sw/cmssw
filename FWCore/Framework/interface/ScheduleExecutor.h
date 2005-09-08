#ifndef Framework_ScheduleExecutor_h
#define Framework_ScheduleExecutor_h

/**
   \file
   Declaration of class ScheduleExecutor

   \author Stefano ARGIRO
   \version $Id: ScheduleExecutor.h,v 1.5 2005/09/01 23:30:49 wmtan Exp $
   \date 18 May 2005
*/

#include <list>
#include "boost/signal.hpp"

namespace edm {
 
  class EventPrincipal;
  class EventSetup;
  class Worker;
  class ActionTable;
  class ModuleDescription;

  /**
     \class ScheduleExecutor ScheduleExecutor.h "edm/ScheduleExecutor.h"

     \brief The class that actually runs the workers

     \author Stefano ARGIRO
     \date 18 May 2005
  */
  class ScheduleExecutor {

  public:
    
    typedef std::list<std::list<Worker* > > PathList;
    typedef std::list<Worker* >  WorkerList;
    //ScheduleExecutor(){}
    ScheduleExecutor(const PathList& pathlist, const ActionTable& actions);

    /// pass @param event to all the workers for them to do the work
    /** return 0 on success (need to define codes) */
    int  runOneEvent(EventPrincipal& eventPrincipal, EventSetup const& eventSetup);

    boost::signal<void (const ModuleDescription&)> preModuleSignal;
    boost::signal<void (const ModuleDescription&)> postModuleSignal;

  private:
    PathList m_pathlist;
    const ActionTable* m_actions;

  }; // ScheduleExecutor


} // edm


#endif
