/**
   \file
   Declaration of class ScheduleBuilder

   \author Stefano ARGIRO
   \version $Id: ScheduleBuilder.h,v 1.2 2005/06/24 14:55:07 argiro Exp $
   \date 18 May 2005
*/

#ifndef _edm_ScheduleBuilder_h_
#define _edm_ScheduleBuilder_h_

static const char CVSId_edm_ScheduleBuilder[] = 
"$Id: ScheduleBuilder.h,v 1.2 2005/06/24 14:55:07 argiro Exp $";

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <list>

namespace edm {

  class Worker;
  class WorkerRegistry;
  class ParameterSet;

  /**
     \class ScheduleBuilder ScheduleBuilder.h "edm/ScheduleBuilder.h"

     \brief Build and check the schedule specified in the ProcessDesc

     \author Stefano ARGIRO
     \date 18 May 2005
  */
  class ScheduleBuilder {

  public:
    ScheduleBuilder(ParameterSet const& processDesc,
		    WorkerRegistry * registry);
    
    typedef std::list<Worker*>    WorkerList;
    typedef std::list<WorkerList> PathList;


    /// Return the list of paths to be executed
    const PathList& getPathList() const;
    

  private:

    const ParameterSet  m_processDesc;
    WorkerRegistry *    m_registry;
    PathList            m_pathList;
  };
} // edm


#endif // _edm_ScheduleBuilder_h_

