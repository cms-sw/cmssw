#include "DQM/EcalMonitorTasks/interface/DQWorkerTask.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace ecaldqm {
  DQWorkerTask::DQWorkerTask() : DQWorker() {}

  /*static*/
  void DQWorkerTask::fillDescriptions(edm::ParameterSetDescription& _desc) { DQWorker::fillDescriptions(_desc); }

  void DQWorkerTask::setME(edm::ParameterSet const& _ps) {
    DQWorker::setME(_ps);

    for (MESetCollection::iterator mItr(MEs_.begin()); mItr != MEs_.end(); ++mItr) {
      if (willConvertToEDM_)
        mItr->second->setBatchMode();
    }
  }

  void DependencySet::formSequenceFragment_(Dependency const& _d,
                                            std::vector<Collections>& _sequence,
                                            std::vector<Collections>::iterator _maxPos) const {
    Collections col(_d.dependant);
    std::vector<Collections>::iterator pos(std::find(_sequence.begin(), _sequence.end(), col));
    if (pos == _sequence.end())
      _sequence.insert(_maxPos, col);
    else if (pos < _maxPos)
      return;
    else
      throw cms::Exception("InvalidConfiguration") << "Circular dependency of collections";

    for (auto rItr : _d.requisite) {
      for (const auto& dItr : set_) {
        if (dItr.dependant != rItr)
          continue;
        pos = std::find(_sequence.begin(), _sequence.end(), col);
        formSequenceFragment_(dItr, _sequence, pos);
        break;
      }
    }
  }

}  // namespace ecaldqm
