#include "DQM/EcalCommon/interface/DQWorker.h"

#include "DQM/EcalCommon/interface/MESetUtils.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/Provenance/interface/EventID.h"

namespace ecaldqm
{
  DQWorker::DQWorker() :
    name_(""),
    MEs_(),
    timestamp_(),
    verbosity_(0),
    onlineMode_(false),
    willConvertToEDM_(true)
  {
  }

  DQWorker::~DQWorker()
  {
  }

  /*static*/
  void
  DQWorker::fillDescriptions(edm::ParameterSetDescription& _desc)
  {
    _desc.addUntracked<bool>("onlineMode", false);
    _desc.addUntracked<bool>("willConvertToEDM", true);

    edm::ParameterSetDescription meParameters;
    edm::ParameterSetDescription meNodeParameters;
    fillMESetDescriptions(meNodeParameters);
    meParameters.addNode(edm::ParameterWildcard<edm::ParameterSetDescription>("*", edm::RequireZeroOrMore, false, meNodeParameters));
    _desc.addUntracked("MEs", meParameters);

    edm::ParameterSetDescription workerParameters;
    workerParameters.setUnknown();
    _desc.addUntracked("params", workerParameters);
  }

  void
  DQWorker::initialize(std::string const& _name, edm::ParameterSet const& _commonParams)
  {
    name_ = _name;
    onlineMode_ = _commonParams.getUntrackedParameter<bool>("onlineMode");
    willConvertToEDM_ = _commonParams.getUntrackedParameter<bool>("willConvertToEDM");
  }

  void
  DQWorker::setME(edm::ParameterSet const& _meParams)
  {
    std::vector<std::string> const& MENames(_meParams.getParameterNames());

    for(unsigned iME(0); iME != MENames.size(); iME++){
      std::string name(MENames[iME]);
      edm::ParameterSet const& params(_meParams.getUntrackedParameterSet(name));

      if(!onlineMode_ && params.getUntrackedParameter<bool>("online")) continue;

      try{
        MEs_.insert(name, createMESet(params));
      }
      catch(std::exception&){
        edm::LogError("EcalDQM") << "Exception caught while constructing MESet " << name;
        throw;
      }
    }
  }

  void
  DQWorker::releaseMEs()
  {
    for(MESetCollection::iterator mItr(MEs_.begin()); mItr != MEs_.end(); ++mItr)
      mItr->second->clear();
  }

  void
  DQWorker::bookMEs(DQMStore& _booker)
  {
    for(MESetCollection::iterator mItr(MEs_.begin()); mItr != MEs_.end(); ++mItr){
      MESet* me(mItr->second);
      if(me->isActive()) continue;
      me->book(_booker);
    }
  }

  void
  DQWorker::bookMEs(DQMStore::IBooker& _booker)
  {
    for(MESetCollection::iterator mItr(MEs_.begin()); mItr != MEs_.end(); ++mItr){
      MESet* me(mItr->second);
      if(me->isActive()) continue;
      me->book(_booker);
    }
  }

  void
  DQWorker::print_(std::string const& _message, int _threshold/* = 0*/) const
  {
    if(verbosity_ > _threshold)
      edm::LogInfo("EcalDQM") << name_ << ": " << _message;
  }


  DQWorker*
  WorkerFactoryStore::getWorker(std::string const& _name, int _verbosity, edm::ParameterSet const& _commonParams, edm::ParameterSet const& _workerParams) const
  {
    DQWorker* worker(workerFactories_.at(_name)());
    worker->setVerbosity(_verbosity);
    worker->initialize(_name, _commonParams);
    worker->setME(_workerParams.getUntrackedParameterSet("MEs"));
    if(_workerParams.existsAs<edm::ParameterSet>("sources", false))
      worker->setSource(_workerParams.getUntrackedParameterSet("sources"));
    if(_workerParams.existsAs<edm::ParameterSet>("params", false))
      worker->setParams(_workerParams.getUntrackedParameterSet("params"));
    return worker;
  }

  /*static*/
  WorkerFactoryStore*
  WorkerFactoryStore::singleton()
  {
    static WorkerFactoryStore workerFactoryStore;
    return &workerFactoryStore;
  }

}

