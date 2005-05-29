/**
   \file
   Declaration of class ScheduleExecutor

   \author Stefano ARGIRO
   \version $Id: ScheduleExecutor.h,v 1.3 2005/05/25 16:14:46 argiro Exp $
   \date 18 May 2005
*/

#ifndef _edm_ScheduleExecutor_h_
#define _edm_ScheduleExecutor_h_

static const char CVSId_edm_ScheduleExecutor[] = 
"$Id: ScheduleExecutor.h,v 1.3 2005/05/25 16:14:46 argiro Exp $";

#include <list>

namespace edm {
 
  class EventPrincipal;
  class EventSetup;
  class Worker;

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

    ScheduleExecutor(const PathList& pathlist);

    /// pass @param event to all the workers for them to do the work
    /** return 0 on success ( need to define codes) */
    int  runOneEvent(EventPrincipal& eventPrincipal, EventSetup const& eventSetup);

  private:
    const PathList m_pathlist;

  }; // ScheduleExecutor


} // edm


#endif // _edm_ScheduleExecutor_h_
