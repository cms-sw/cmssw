/**
   \file
   Declaration of class ScheduleExecutor

   \author Stefano ARGIRO
   \version $Id: ScheduleExecutor.h,v 1.3 2005/06/23 19:59:30 wmtan Exp $
   \date 18 May 2005
*/

#ifndef _edm_ScheduleExecutor_h_
#define _edm_ScheduleExecutor_h_

static const char CVSId_edm_ScheduleExecutor[] = 
"$Id: ScheduleExecutor.h,v 1.3 2005/06/23 19:59:30 wmtan Exp $";

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


#endif // _edm_ScheduleExecutor_h_
