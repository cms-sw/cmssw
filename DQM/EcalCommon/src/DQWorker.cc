#include "DQM/EcalCommon/interface/DQWorker.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

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

  std::map<std::string, std::map<std::string, unsigned> > DQWorker::meOrderingMaps;

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

    string topDir;
    if(_workerParams.existsAs<string>("topDirectory", false))
      topDir = _workerParams.getUntrackedParameter<string>("topDirectory");
    else
      topDir = _commonParams.getUntrackedParameter<string>("topDirectory");

    edm::ParameterSet const& MEParams(_workerParams.getUntrackedParameterSet("MEs"));
    vector<string> const& MENames(MEParams.getParameterNames());

    MEs_.resize(MENames.size());

    map<string, unsigned> const& nameToIndex(mItr->second);

    for(unsigned iME(0); iME < MENames.size(); iME++){
      string const& MEName(MENames[iME]);

      map<string, unsigned>::const_iterator nItr(nameToIndex.find(MEName));
      if(nItr == nameToIndex.end())
        throw cms::Exception("InvalidConfiguration") << "Cannot find ME index for " << MEName;

      MESet* meSet(createMESet_(topDir, MEParams.getUntrackedParameterSet(MEName)));
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
  DQWorker::setMEOrdering(std::map<std::string, unsigned>&)
  {
  }

  MESet*
  DQWorker::createMESet_(std::string const& _topDir, edm::ParameterSet const& _MEParam) const
  {
    BinService const* binService(&(*(edm::Service<EcalDQMBinningService>())));
    if(!binService)
      throw cms::Exception("Service") << "EcalDQMBinningService not found" << std::endl;

    std::string fullPath(_topDir + "/" + _MEParam.getUntrackedParameter<std::string>("path"));
    BinService::ObjectType otype(binService->getObjectType(_MEParam.getUntrackedParameter<std::string>("otype")));
    BinService::BinningType btype(binService->getBinningType(_MEParam.getUntrackedParameter<std::string>("btype")));
    MonitorElement::Kind kind(MESet::translateKind(_MEParam.getUntrackedParameter<std::string>("kind")));

    BinService::AxisSpecs const* xaxis(0);
    BinService::AxisSpecs const* yaxis(0);
    BinService::AxisSpecs const* zaxis(0);
    if(_MEParam.existsAs<edm::ParameterSet>("xaxis", false)) xaxis = binService->formAxis(_MEParam.getUntrackedParameterSet("xaxis"));
    if(_MEParam.existsAs<edm::ParameterSet>("yaxis", false)) yaxis = binService->formAxis(_MEParam.getUntrackedParameterSet("yaxis"));
    if(_MEParam.existsAs<edm::ParameterSet>("zaxis", false)) zaxis = binService->formAxis(_MEParam.getUntrackedParameterSet("zaxis"));

    if(otype == BinService::nObjType)
      return new MESetNonObject(fullPath, otype, btype, kind, xaxis, yaxis, zaxis);

    if(otype == BinService::kChannel)
      return new MESetChannel(fullPath, otype, btype, kind);

    if(btype == BinService::kProjEta || btype == BinService::kProjPhi)
      return new MESetProjection(fullPath, otype, btype, kind, yaxis);

    if(btype == BinService::kTrend)
      return new MESetTrend(fullPath, otype, btype, kind, yaxis);

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
    if(logicalDimensions == 2 && yaxis && btype != BinService::kUser) logicalDimensions = 1;

    // for EventInfo summary contents
    if(btype == BinService::kReport){
      if(logicalDimensions != 0)
	throw cms::Exception("InvalidCall") << "Report can only be a DQM_KIND_REAL" << std::endl;
    }

    if(btype == BinService::kUser)
      return new MESetEcal(fullPath, otype, btype, kind, logicalDimensions, xaxis, yaxis, zaxis);

    if(logicalDimensions == 0)
      return new MESetDet0D(fullPath, otype, btype, kind);

    if(logicalDimensions == 1)
      return new MESetDet1D(fullPath, otype, btype, kind, yaxis);

    if(logicalDimensions == 2)
      return new MESetDet2D(fullPath, otype, btype, kind, zaxis);

    return 0;
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

