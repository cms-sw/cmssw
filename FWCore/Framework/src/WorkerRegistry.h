#ifndef Framework_WorkerRegistry_h
#define Framework_WorkerRegistry_h

/**
   \file
   Declaration of class ModuleRegistry

   \author Stefano ARGIRO
   \date 18 May 2005
*/

#include <memory>

#include <map>
#include <string>


namespace edm {

  class Worker;
  class ActivityRegistry;
  struct WorkerParams;
  class ModuleRegistry;
  class ParameterSet;
  namespace maker {
    class ModuleHolder;
  }

  /**
     \class WorkerRegistry WorkerRegistry.h "edm/WorkerRegistry.h"

     \brief The Registry of all workers that where requested
     Holds all instances of workers. In this implementation, Workers 
     are owned.
  */

  class WorkerRegistry {

  public:

    explicit WorkerRegistry(std::shared_ptr<ActivityRegistry> areg);
    WorkerRegistry(std::shared_ptr<ActivityRegistry> areg,
                   std::shared_ptr<ModuleRegistry> iModReg);
    ~WorkerRegistry();
        
    WorkerRegistry(WorkerRegistry const&) = delete; // Disallow copying and moving
    WorkerRegistry& operator=(WorkerRegistry const&) = delete; // Disallow copying and moving

    /// Retrieve the particular instance of the worker
    /** If the worker with that set of parameters does not exist,
        create it
        @note Workers are owned by this class, do not delete them*/
    Worker* getWorker(WorkerParams const& p, std::string const& moduleLabel);
    void clear();
    
  private:
    /// the container of workers
    typedef std::map<std::string, std::shared_ptr<Worker> > WorkerMap;

    std::shared_ptr<ModuleRegistry> modRegistry_;
    
    /// internal map of registered workers (owned).
    WorkerMap m_workerMap;
    std::shared_ptr<ActivityRegistry> actReg_;
     
  }; // WorkerRegistry


} // edm


#endif
