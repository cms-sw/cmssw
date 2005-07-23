/**
   \file
   Declaration of class ScheduleBuilder

   \author Stefano ARGIRO
   \version $Id: ScheduleBuilder.h,v 1.5 2005/07/21 20:49:33 argiro Exp $
   \date 18 May 2005
*/

#ifndef _edm_ScheduleBuilder_h_
#define _edm_ScheduleBuilder_h_

static const char CVSId_edm_ScheduleBuilder[] = 
"$Id: ScheduleBuilder.h,v 1.5 2005/07/21 20:49:33 argiro Exp $";

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <list>

namespace edm {

  class Worker;
  class WorkerRegistry;
  class ProductRegistry;
  class ParameterSet;
  class ActionTable;

  /**
     \class ScheduleBuilder ScheduleBuilder.h "edm/ScheduleBuilder.h"

     \brief Build and check the schedule specified in the ProcessDesc

     \author Stefano ARGIRO
     \date 18 May 2005
  */
  class ScheduleBuilder {

  public:
    ScheduleBuilder(ParameterSet const& processDesc,
		    WorkerRegistry& wregistry,
		    ProductRegistry& pregistry,
		    ActionTable& actions);
    
    typedef std::list<Worker*>    WorkerList;
    typedef std::list<WorkerList> PathList;


    /// Return the list of paths to be executed
    const PathList& getPathList() const;
    

  private:

    const ParameterSet  m_processDesc;
    PathList            m_pathList;
  };
} // edm


#endif // _edm_ScheduleBuilder_h_

