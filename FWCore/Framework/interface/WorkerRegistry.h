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

#include "FWCore/Utilities/interface/propagate_const.h"

namespace edm {

  class Worker;
  class ActivityRegistry;
  class ModuleRegistry;
  class ExceptionToActionTable;
  /**
     \class WorkerRegistry WorkerRegistry.h "edm/WorkerRegistry.h"

     \brief The Registry of all workers that were requested
     Holds all instances of workers. In this implementation, Workers 
     are owned.
  */

  class WorkerRegistry {
  public:
    WorkerRegistry(std::shared_ptr<ActivityRegistry> areg, std::shared_ptr<ModuleRegistry> iModReg);
    ~WorkerRegistry();

    WorkerRegistry(WorkerRegistry&&) = default;
    WorkerRegistry(WorkerRegistry const&) = delete;             // Disallow copying and moving
    WorkerRegistry& operator=(WorkerRegistry const&) = delete;  // Disallow copying and moving

    /// Retrieve particular instance of the worker without creating it
    /// If one doesn't exist, returns nullptr
    Worker const* get(std::string const& moduleLabel) const;

    /** Creates worker if doesn't already exist
     * @note Workers are owned by this class, do not delete them
     */
    Worker* getWorkerFromExistingModule(std::string const& moduleLabel, ExceptionToActionTable const* actions);

    /// Deletes the module of the Worker, but the Worker continues to exist.
    void deleteModule(std::string const& moduleLabel);

    void clear();

  private:
    /// the container of workers
    typedef std::map<std::string, edm::propagate_const<std::shared_ptr<Worker>>> WorkerMap;

    edm::propagate_const<std::shared_ptr<ModuleRegistry>> modRegistry_;

    /// internal map of registered workers (owned).
    WorkerMap m_workerMap;
    std::shared_ptr<ActivityRegistry> actReg_;  // We do not use propagate_const because the registry itself is mutable.

  };  // WorkerRegistry

}  // namespace edm

#endif
