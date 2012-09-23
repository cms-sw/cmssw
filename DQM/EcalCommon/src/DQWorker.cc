#include "DQM/EcalCommon/interface/DQWorker.h"

#include "DQM/EcalCommon/interface/MESetUtils.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Provenance/interface/EventID.h"

namespace ecaldqm{

  std::map<std::string, std::map<std::string, unsigned> > DQWorker::meOrderingMaps;
  bool DQWorker::online(false);
  time_t DQWorker::now(0);
  edm::RunNumber_t DQWorker::iRun(0);
  edm::LuminosityBlockNumber_t DQWorker::iLumi(0);
  edm::EventNumber_t DQWorker::iEvt(0);

  DQWorker::DQWorker(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams, std::string const& _name) :
    name_(_name),
    MEs_(0),
    initialized_(false),
    verbosity_(0)
  {
    using namespace std;

    map<string, map<string, unsigned> >::const_iterator mItr(meOrderingMaps.find(name_));
    if(mItr == meOrderingMaps.end())
      throw cms::Exception("InvalidConfiguration") << "Cannot find ME ordering for " << name_;

    BinService const* binService(&(*(edm::Service<EcalDQMBinningService>())));
    if(!binService)
      throw cms::Exception("Service") << "EcalDQMBinningService not found" << std::endl;

    edm::ParameterSet const& MEParams(_workerParams.getUntrackedParameterSet("MEs"));
    vector<string> const& MENames(MEParams.getParameterNames());

    MEs_.resize(MENames.size());

    map<string, unsigned> const& nameToIndex(mItr->second);

    for(unsigned iME(0); iME < MENames.size(); iME++){
      string const& MEName(MENames[iME]);

      map<string, unsigned>::const_iterator nItr(nameToIndex.find(MEName));
      if(nItr == nameToIndex.end())
        throw cms::Exception("InvalidConfiguration") << "Cannot find ME index for " << MEName;

      MESet* meSet(createMESet(MEParams.getUntrackedParameterSet(MEName), binService));
      if(meSet) MEs_[nItr->second] = meSet;
    }
  }

  DQWorker::~DQWorker()
  {
    for(unsigned iME(0); iME < MEs_.size(); iME++)
      delete MEs_[iME];
  }

  void
  DQWorker::bookMEs()
  {
    for(unsigned iME(0); iME < MEs_.size(); iME++){
      if(MEs_[iME]){
        if(MEs_[iME]->getBinType() == BinService::kTrend && !online) continue;
        if(MEs_[iME]->isActive()) continue;
        MEs_[iME]->book();
      }
    }
  }

  void
  DQWorker::reset()
  {
    for(unsigned iME(0); iME < MEs_.size(); iME++)
      if(MEs_[iME]) MEs_[iME]->clear();

    initialized_ = false;
  }

  void
  DQWorker::initialize()
  {
    initialized_ = true;
  }

  /*static*/
  void
  DQWorker::setMEOrdering(std::map<std::string, unsigned>&)
  {
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

