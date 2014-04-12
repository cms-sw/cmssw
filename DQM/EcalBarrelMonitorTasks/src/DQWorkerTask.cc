#include "../interface/DQWorkerTask.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace ecaldqm
{
  DQWorkerTask::DQWorkerTask() :
    DQWorker(),
    resettable_()
  {
  }

  /*static*/
  void
  DQWorkerTask::fillDescriptions(edm::ParameterSetDescription& _desc)
  {
    DQWorker::fillDescriptions(_desc);
  }

  void
  DQWorkerTask::setME(edm::ParameterSet const& _ps)
  {
    DQWorker::setME(_ps);
    
    for(MESetCollection::iterator mItr(MEs_.begin()); mItr != MEs_.end(); ++mItr){
      if(willConvertToEDM_) mItr->second->setBatchMode();
      
      // TEMPORARY MEASURE - softReset does not accept variable bin size as of September 2012
      // isVariableBinning is true for 1. MESetEcal or MESetNonObject with any custom binning or 2. MESetTrend
      // In principle it is sufficient to protect the MESetTrends from being reset
      if(/*mItr->second->getBinType() != BinService::kTrend &&*/
         !mItr->second->isVariableBinning() &&
         mItr->second->getKind() != MonitorElement::DQM_KIND_REAL)
        resettable_.insert(mItr->first);
    }
  }

  void
  DQWorkerTask::softReset()
  {
    std::for_each(resettable_.begin(), resettable_.end(), [this](std::string const& name){
        this->MEs_.at(name).softReset();
      });
  }

  void
  DQWorkerTask::recoverStats()
  {
    std::for_each(resettable_.begin(), resettable_.end(), [this](std::string const& name){
        this->MEs_.at(name).recoverStats();
      });
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

