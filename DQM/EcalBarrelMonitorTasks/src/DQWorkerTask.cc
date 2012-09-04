#include "../interface/DQWorkerTask.h"

#include "FWCore/Utilities/interface/Exception.h"

namespace ecaldqm {

  DQWorkerTask::DQWorkerTask(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams, std::string const& _name) :
    DQWorker(_workerParams, _commonParams, _name),
    collectionMask_(0)
  {
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

  void
  DependencySet::formSequenceFragment_(Dependency const& _d, std::vector<Collections>& _sequence, std::vector<Collections>::iterator _maxPos) const
  {
    Collections col(_d.dependant);
    std::vector<Collections>::iterator pos(std::find(_sequence.begin(), _sequence.end(), col));
    if(pos == _sequence.end()) _sequence.insert(_maxPos, col);
    else if(pos < _maxPos) return;
    else
      throw cms::Exception("InvalidConfiguration") << "Circular dependency of collections";

    for(std::set<Collections>::const_iterator rItr(_d.requisite.begin()); rItr != _d.requisite.end(); ++rItr){
      for(std::vector<Dependency>::const_iterator dItr(set_.begin()); dItr != set_.end(); ++dItr){
        if(dItr->dependant != *rItr) continue;
        pos = std::find(_sequence.begin(), _sequence.end(), col);
        formSequenceFragment_(*dItr, _sequence, pos);
        break;
      }
    }
  }

}

