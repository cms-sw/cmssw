#include "DQM/EcalCommon/interface/DQWorker.h"

#include "DQM/EcalCommon/interface/MESetUtils.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Provenance/interface/EventID.h"

namespace ecaldqm{

  bool DQWorker::online(false);
  time_t DQWorker::now(0);
  edm::RunNumber_t DQWorker::iRun(0);
  edm::LuminosityBlockNumber_t DQWorker::iLumi(0);
  edm::EventNumber_t DQWorker::iEvt(0);

  DQWorker::DQWorker(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams, std::string const& _name) :
    name_(_name),
    MEs_(),
    initialized_(false),
    verbosity_(0)
  {
    using namespace std;

    edm::ParameterSet const& MEParams(_workerParams.getUntrackedParameterSet("MEs"));
    vector<string> const& MENames(MEParams.getParameterNames());

    for(unsigned iME(0); iME < MENames.size(); iME++){
      string const& MEName(MENames[iME]);
      MEs_.insert(MEName, createMESet(MEParams.getUntrackedParameterSet(MEName)));
    }
  }

  DQWorker::~DQWorker()
  {
  }

  void
  DQWorker::bookMEs()
  {
    for(MESetCollection::iterator mItr(MEs_.begin()); mItr != MEs_.end(); ++mItr){
      MESet* me(mItr->second);
      if(me){
        if(me->getBinType() == BinService::kTrend && !online) continue;
        if(me->isActive()) continue;
        me->book();
      }
    }
  }

  void
  DQWorker::reset()
  {
    for(MESetCollection::iterator mItr(MEs_.begin()); mItr != MEs_.end(); ++mItr)
      if(mItr->second) mItr->second->clear();

    initialized_ = false;
  }

  void
  DQWorker::initialize()
  {
    initialized_ = true;
  }

  void
  DQWorker::print_(std::string const& _message, int _threshold/* = 0*/) const
  {
    if(verbosity_ > _threshold)
      std::cout << name_ << ": " << _message << std::endl;
  }



  std::map<std::string, WorkerFactory> WorkerFactoryHelper::workerFactories_;

  WorkerFactory
  WorkerFactoryHelper::findFactory(const std::string &_name)
  {
    if(workerFactories_.find(_name) != workerFactories_.end()) return workerFactories_[_name];
    return NULL;
  }

}

