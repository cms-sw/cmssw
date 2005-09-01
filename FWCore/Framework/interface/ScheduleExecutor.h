#ifndef Framework_ScheduleExecutor_h
#define Framework_ScheduleExecutor_h

/**
   \file
   Declaration of class ScheduleExecutor

   \author Stefano ARGIRO
   \version $Id: ScheduleExecutor.h,v 1.4 2005/07/20 03:00:36 jbk Exp $
   \date 18 May 2005
*/

#include <list>

namespace edm {
 
  class EventPrincipal;
  class EventSetup;
  class Worker;
  class ActionTable;

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
    ScheduleExecutor(){}
    ScheduleExecutor(const PathList& pathlist, const ActionTable& actions);

    /// pass @param event to all the workers for them to do the work
    /** return 0 on success (need to define codes) */
    int  runOneEvent(EventPrincipal& eventPrincipal, EventSetup const& eventSetup);

  private:
    PathList m_pathlist;
    const ActionTable* m_actions;

  }; // ScheduleExecutor


} // edm


#endif
