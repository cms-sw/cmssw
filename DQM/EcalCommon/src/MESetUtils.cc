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

#include "TString.h"
#include "TPRegexp.h"

namespace ecaldqm
{
  MESet*
  createMESet(edm::ParameterSet const& _MEParam)
  {
    std::string path(_MEParam.getUntrackedParameter<std::string>("path"));
    BinService::ObjectType otype(BinService::getObjectType(_MEParam.getUntrackedParameter<std::string>("otype")));
    BinService::BinningType btype(BinService::getBinningType(_MEParam.getUntrackedParameter<std::string>("btype")));
    MonitorElement::Kind kind(MESet::translateKind(_MEParam.getUntrackedParameter<std::string>("kind")));

    BinService::AxisSpecs const* xaxis(0);
    BinService::AxisSpecs const* yaxis(0);
    BinService::AxisSpecs const* zaxis(0);
    if(_MEParam.existsAs<edm::ParameterSet>("xaxis", false)) xaxis = BinService::formAxis(_MEParam.getUntrackedParameterSet("xaxis"));
    if(_MEParam.existsAs<edm::ParameterSet>("yaxis", false)) yaxis = BinService::formAxis(_MEParam.getUntrackedParameterSet("yaxis"));
    if(_MEParam.existsAs<edm::ParameterSet>("zaxis", false)) zaxis = BinService::formAxis(_MEParam.getUntrackedParameterSet("zaxis"));

    MESet* set(0);

    if(btype == BinService::kTrend){
      bool minutely(false);
      bool cumulative(false);
      if(_MEParam.existsAs<bool>("minutely", false))
        minutely = _MEParam.getUntrackedParameter<bool>("minutely");
      if(_MEParam.existsAs<bool>("cumulative", false))
        cumulative = _MEParam.getUntrackedParameter<bool>("cumulative");
      set  = new MESetTrend(path, otype, btype, kind, minutely, cumulative, yaxis);
    }
    else if(otype == BinService::nObjType)
      set = new MESetNonObject(path, otype, btype, kind, xaxis, yaxis, zaxis);
    else if(otype == BinService::kChannel)
      set = new MESetChannel(path, otype, btype, kind);
    else if(btype == BinService::kProjEta || btype == BinService::kProjPhi)
      set = new MESetProjection(path, otype, btype, kind, yaxis);
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
        throw cms::Exception("InvalidConfiguration") << "Histogram type " << kind << " not supported" << std::endl;
      }

      // example case: Ecal/TriggerPrimitives/EmulMatching/TrigPrimTask matching index
      if(logicalDimensions == 2 && yaxis && btype != BinService::kUser) logicalDimensions = 1;

      // for EventInfo summary contents
      if(btype == BinService::kReport){
        if(logicalDimensions != 0)
          throw cms::Exception("InvalidConfiguration") << "Report can only be a DQM_KIND_REAL" << std::endl;
      }

      if(btype == BinService::kUser)
        set = new MESetEcal(path, otype, btype, kind, logicalDimensions, xaxis, yaxis, zaxis);
      else if(logicalDimensions == 0)
        set = new MESetDet0D(path, otype, btype, kind);
      else if(logicalDimensions == 1)
        set = new MESetDet1D(path, otype, btype, kind, yaxis);
      else if(logicalDimensions == 2)
        set = new MESetDet2D(path, otype, btype, kind, zaxis);
    }

    if(_MEParam.existsAs<int>("multi", false)){
      MESet* tmp(set);
      set = new MESetMulti(*tmp, _MEParam.getUntrackedParameter<int>("multi"));
      delete tmp;
    }

    if(!set) throw cms::Exception("InvalidConfiguration") << "MESet " << path << " could not be initialized";

    return set;
  }

  void
  formPath(std::string& _path, std::map<std::string, std::string> const& _replacements)
  {
    TString path(_path);

    for(std::map<std::string, std::string>::const_iterator repItr(_replacements.begin()); repItr != _replacements.end(); ++repItr){

      TString pattern("\\%\\(");
      pattern += repItr->first;
      pattern += "\\)s";

      TPRegexp re(pattern);

      re.Substitute(path, repItr->second, "g");
    }

    _path = path.Data();
  }


}
