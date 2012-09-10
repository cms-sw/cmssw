#include "DQM/EcalCommon/interface/MESetUtils.h"

#include "DQM/EcalCommon/interface/MESetNonObject.h"
#include "DQM/EcalCommon/interface/MESetChannel.h"
#include "DQM/EcalCommon/interface/MESetEcal.h"
#include "DQM/EcalCommon/interface/MESetDet0D.h"
#include "DQM/EcalCommon/interface/MESetDet1D.h"
#include "DQM/EcalCommon/interface/MESetDet2D.h"
#include "DQM/EcalCommon/interface/MESetProjection.h"
#include "DQM/EcalCommon/interface/MESetTrend.h"
#include "DQM/EcalCommon/interface/MESetMulti.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace ecaldqm
{
  MESet*
  createMESet(edm::ParameterSet const& _MEParam, BinService const* _binService, std::string _topDir/* = ""*/, DQMStore* _dqmStore/* = 0*/)
  {
    std::string fullPath;
    if(_topDir != "")
      fullPath = _topDir + "/" + _MEParam.getUntrackedParameter<std::string>("path");
    else if(_dqmStore){
      _dqmStore->cd();

      std::string path(_MEParam.getUntrackedParameter<std::string>("path"));
      std::string fullPath;

      std::vector<std::string> dirs(_dqmStore->getSubdirs());
      for(unsigned iD(0); iD < dirs.size(); ++iD){
        std::string& topDir(dirs[iD]);
        fullPath = topDir + "/" + path;
        if(_dqmStore->get(fullPath)) break;
        else fullPath = "";
      }
    }

    if(fullPath == "") return 0;

    BinService::ObjectType otype(_binService->getObjectType(_MEParam.getUntrackedParameter<std::string>("otype")));
    BinService::BinningType btype(_binService->getBinningType(_MEParam.getUntrackedParameter<std::string>("btype")));
    MonitorElement::Kind kind(MESet::translateKind(_MEParam.getUntrackedParameter<std::string>("kind")));

    BinService::AxisSpecs const* xaxis(0);
    BinService::AxisSpecs const* yaxis(0);
    BinService::AxisSpecs const* zaxis(0);
    if(_MEParam.existsAs<edm::ParameterSet>("xaxis", false)) xaxis = _binService->formAxis(_MEParam.getUntrackedParameterSet("xaxis"));
    if(_MEParam.existsAs<edm::ParameterSet>("yaxis", false)) yaxis = _binService->formAxis(_MEParam.getUntrackedParameterSet("yaxis"));
    if(_MEParam.existsAs<edm::ParameterSet>("zaxis", false)) zaxis = _binService->formAxis(_MEParam.getUntrackedParameterSet("zaxis"));

    MESet* set(0);

    if(btype == BinService::kTrend){
      bool minutely(false);
      bool cumulative(false);
      if(_MEParam.existsAs<bool>("minutely", false))
        minutely = _MEParam.getUntrackedParameter<bool>("minutely");
      if(_MEParam.existsAs<bool>("cumulative", false))
        cumulative = _MEParam.getUntrackedParameter<bool>("cumulative");
      set  = new MESetTrend(fullPath, otype, btype, kind, minutely, cumulative, yaxis);
    }
    else if(otype == BinService::nObjType)
      set = new MESetNonObject(fullPath, otype, btype, kind, xaxis, yaxis, zaxis);
    else if(otype == BinService::kChannel)
      set = new MESetChannel(fullPath, otype, btype, kind);
    else if(btype == BinService::kProjEta || btype == BinService::kProjPhi)
      set = new MESetProjection(fullPath, otype, btype, kind, yaxis);
    else{
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
        set = new MESetEcal(fullPath, otype, btype, kind, logicalDimensions, xaxis, yaxis, zaxis);
      else if(logicalDimensions == 0)
        set = new MESetDet0D(fullPath, otype, btype, kind);
      else if(logicalDimensions == 1)
        set = new MESetDet1D(fullPath, otype, btype, kind, yaxis);
      else if(logicalDimensions == 2)
        set = new MESetDet2D(fullPath, otype, btype, kind, zaxis);
    }

    if(_MEParam.existsAs<int>("multi", false)){
      MESet* tmp(set);
      set = new MESetMulti(*tmp, _MEParam.getUntrackedParameter<int>("multi"));
      delete tmp;
    }

    return set;
  }

}
