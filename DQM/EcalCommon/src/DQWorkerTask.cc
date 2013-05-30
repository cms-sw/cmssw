#include "DQM/EcalCommon/interface/DQWorkerTask.h"

namespace ecaldqm {

  DQWorkerTask::DQWorkerTask(const edm::ParameterSet& _params, const edm::ParameterSet& _paths, std::string const& _name) :
    DQWorker(_params, _paths, _name),
    collectionMask_(0),
    dependencies_()
  {
  }

  const std::vector<std::pair<Collections, Collections> >&
  DQWorkerTask::getDependencies()
  {
    return dependencies_;
  }

  bool
  DQWorkerTask::runsOn(unsigned _collection)
  {
    if(_collection >= nProcessedObjects) return false;
    return (collectionMask_ >> _collection) & 0x1;
  }

  bool
  DQWorkerTask::filterRunType(const std::vector<short>&)
  {
    return true;
  }

  bool
  DQWorkerTask::filterTrigger(const edm::TriggerResultsByName &)
  {
    return true;
  }

}

