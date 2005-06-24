/**
   \file
   Declaration of class ModuleRegistry

   \author Stefano ARGIRO
   \version $Id: WorkerRegistry.h,v 1.1 2005/05/29 02:29:54 wmtan Exp $
   \date 18 May 2005
*/

#ifndef _edm_ModuleRegistry_h_
#define _edm_ModuleRegistry_h_

static const char CVSId_edm_ModuleRegistry[] = "$Id: WorkerRegistry.h,v 1.1 2005/05/29 02:29:54 wmtan Exp $";

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <map>
#include <string>


namespace edm {

  class Worker;

  /**
     \class ModuleRegistry ModuleRegistry.h "edm/ModuleRegistry.h"

     \brief The Registry of all workers that where requested
     Holds all instances of workers. In this implementation, Workers 
     are owned.

     \author Stefano ARGIRO
     \date 18 May 200
  */

  class WorkerRegistry {

  public:
    
    /// retrieve the single instance
    static WorkerRegistry* get();

    ~WorkerRegistry();
        
    /// Retrieve the particular instance of the worker
    /** If the worker with that set of parameters does not exist,
        create it
        @note Workers are owned by this class, do not delete them*/
    Worker*  getWorker(ParameterSet const& parameterSet,
		       std::string const& processName,
		       unsigned long versionNumber,
		       unsigned long pass);
    
  private:
     
    WorkerRegistry(){}
  
    /// Get a unique name for the worker
    /** Form a string to be used as a key in the map of workers */
    std::string mangleWorkerParameters(ParameterSet const& parameterSet,
				       std::string const& processName,
				       unsigned long versionNumber,
				       unsigned long pass);

    /// the container of workers
    typedef std::map<std::string, Worker*> WorkerMap;

    /// internal map of registered workers (owned). 
    WorkerMap m_workerMap;
     
  }; // WorkerRegistry


} // edm


#endif // _edm_WorkerRegistry_h_

