#include "DQM/EcalCommon/interface/DQWorker.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DQM/EcalCommon/interface/MESet.h"
#include "DQM/EcalCommon/interface/MESetNonObject.h"
#include "DQM/EcalCommon/interface/MESetChannel.h"
#include "DQM/EcalCommon/interface/MESetEcal.h"
#include "DQM/EcalCommon/interface/MESetDet0D.h"
#include "DQM/EcalCommon/interface/MESetDet1D.h"
#include "DQM/EcalCommon/interface/MESetDet2D.h"
#include "DQM/EcalCommon/interface/MESetTrend.h"

namespace ecaldqm{

  std::map<std::string, std::vector<MEData> > DQWorker::meData;

  DQWorker::DQWorker(const edm::ParameterSet &, const edm::ParameterSet& _paths, std::string const& _name) :
    name_(_name),
    MEs_(0),
    initialized_(false),
    verbosity_(0)
  {
    using namespace std;

    map<string, vector<MEData> >::iterator dItr(meData.find(name_));
    if(dItr == meData.end())
      throw cms::Exception("InvalidCall") << "MonitorElement setup data not found for " << name_ << std::endl;

    vector<MEData> const& vData(dItr->second);
    MEs_.resize(vData.size());

    for(unsigned iME(0); iME < MEs_.size(); iME++){
      MEData& data(meData[name_].at(iME));
      string fullpath(_paths.getUntrackedParameter<string>(data.pathName));

      MEs_.at(iME) = createMESet_(fullpath, data);
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
    for(unsigned iME(0); iME < MEs_.size(); iME++)
      if(MEs_[iME]) MEs_[iME]->book();
  }

  void
  DQWorker::reset()
  {
    for(unsigned iME(0); iME < MEs_.size(); iME++)
      if(MEs_[iME]) MEs_[iME]->clear();

    initialized_ = false;
  }

  /*static*/
  void
  DQWorker::setMEData(std::vector<MEData>&)
  {
  }

  MESet*
  DQWorker::createMESet_(std::string const& _fullpath, MEData const& _data, bool _readOnly/* = false*/) const
  {
    BinService::ObjectType otype(_data.otype);
    BinService::BinningType btype(_data.btype);
    MonitorElement::Kind kind(_data.kind);

    if(otype == BinService::nObjType)
      return new MESetNonObject(_fullpath, _data, _readOnly);

    if(otype == BinService::kChannel)
      return new MESetChannel(_fullpath, _data, _readOnly);

    if(btype == BinService::kTrend)
      return new MESetTrend(_fullpath, _data, _readOnly);

    unsigned logicalDimensions;
    switch(kind){
    case MonitorElement::DQM_KIND_REAL:
      logicalDimensions = 0;
      break;
    case MonitorElement::DQM_KIND_TH1F:
    case MonitorElement::DQM_KIND_TPROFILE:
      logicalDimensions = 1;
      break;
    case MonitorElement::DQM_KIND_TH2F:
    case MonitorElement::DQM_KIND_TPROFILE2D:
      logicalDimensions = 2;
      break;
    default:
      throw cms::Exception("InvalidCall") << "Histogram type " << kind << " not supported" << std::endl;
    }

    // example case: Ecal/TriggerPrimitives/EmulMatching/TrigPrimTask matching index
    if(logicalDimensions == 2 && _data.yaxis && btype != BinService::kUser) logicalDimensions = 1;

    // for EventInfo summary contents
    if(btype == BinService::kReport){
      if(logicalDimensions != 0)
	throw cms::Exception("InvalidCall") << "Report can only be a DQM_KIND_REAL" << std::endl;
    }

    if(btype == BinService::kUser)
      return new MESetEcal(_fullpath, _data, logicalDimensions, _readOnly);

    if(logicalDimensions == 0)
      return new MESetDet0D(_fullpath, _data, _readOnly);

    if(logicalDimensions == 1)
      return new MESetDet1D(_fullpath, _data, _readOnly);

    if(logicalDimensions == 2)
      return new MESetDet2D(_fullpath, _data, _readOnly);

    return 0;
  }



  std::map<std::string, WorkerFactory> SetWorker::workerFactories_;

  WorkerFactory
  SetWorker::findFactory(const std::string &_name)
  {
    if(workerFactories_.find(_name) != workerFactories_.end()) return workerFactories_[_name];
    return NULL;
  }

}

