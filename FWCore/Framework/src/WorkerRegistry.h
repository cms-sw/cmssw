#ifndef Framework_WorkerRegistry_h
#define Framework_WorkerRegistry_h

/**
   \file
   Declaration of class ModuleRegistry

   \author Stefano ARGIRO
   \date 18 May 2005
*/

#include "boost/shared_ptr.hpp"
#include "boost/utility.hpp"

#include <map>
#include <string>


namespace edm {

  class Worker;
  struct ActivityRegistry;
  struct WorkerParams;

  /**
     \class ModuleRegistry ModuleRegistry.h "edm/ModuleRegistry.h"

     \brief The Registry of all workers that where requested
     Holds all instances of workers. In this implementation, Workers 
     are owned.
  */

  class WorkerRegistry : private boost::noncopyable {

  public:

    explicit WorkerRegistry(boost::shared_ptr<ActivityRegistry> areg);
    ~WorkerRegistry();
        
    /// Retrieve the particular instance of the worker
    /** If the worker with that set of parameters does not exist,
        create it
        @note Workers are owned by this class, do not delete them*/
    Worker* getWorker(WorkerParams const& p, std::string const& moduleLabel);
    void clear();
    
  private:
    /// the container of workers
    typedef std::map<std::string, boost::shared_ptr<Worker> > WorkerMap;

    /// internal map of registered workers (owned). 
    WorkerMap m_workerMap;
    boost::shared_ptr<ActivityRegistry> actReg_;
     
  }; // WorkerRegistry


} // edm


#endif
