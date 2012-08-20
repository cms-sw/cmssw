#include "DQM/EcalCommon/interface/DQWorker.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DQM/EcalCommon/interface/MESet.h"
#include "DQM/EcalCommon/interface/MESetNonObject.h"
#include "DQM/EcalCommon/interface/MESetChannel.h"
#include "DQM/EcalCommon/interface/MESetEcal.h"
#include "DQM/EcalCommon/interface/MESetDet0D.h"
#include "DQM/EcalCommon/interface/MESetDet1D.h"
#include "DQM/EcalCommon/interface/MESetDet2D.h"
#include "DQM/EcalCommon/interface/MESetProjection.h"
#include "DQM/EcalCommon/interface/MESetTrend.h"

namespace ecaldqm{

  std::map<std::string, std::vector<MEData> > DQWorker::meData;

  DQWorker::DQWorker(const edm::ParameterSet& _params, std::string const& _name) :
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

    for(unsigned iME(0); iME < vData.size(); iME++){
      MEData const& data(vData[iME]);
      if(data.kind == MonitorElement::DQM_KIND_INVALID) continue;

      MEs_.push_back(createMESet_(data));
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

  void
  DQWorker::initialize()
  {
    initialized_ = true;
  }

  /*static*/
  void
  DQWorker::setMEData(std::vector<MEData>&)
  {
  }

  MESet*
  DQWorker::createMESet_(MEData const& _data) const
  {
    BinService::ObjectType otype(_data.otype);
    BinService::BinningType btype(_data.btype);
    MonitorElement::Kind kind(_data.kind);

    if(otype == BinService::nObjType)
      return new MESetNonObject(_data);

    if(otype == BinService::kChannel)
      return new MESetChannel(_data);

    if(btype == BinService::kProjEta || btype == BinService::kProjPhi)
      return new MESetProjection(_data);

    if(btype == BinService::kTrend)
      return new MESetTrend(_data);

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
      return new MESetEcal(_data, logicalDimensions);

    if(logicalDimensions == 0)
      return new MESetDet0D(_data);

    if(logicalDimensions == 1)
      return new MESetDet1D(_data);

    if(logicalDimensions == 2)
      return new MESetDet2D(_data);

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

