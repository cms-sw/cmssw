#include "DQM/EcalCommon/interface/MESetUtils.h"

#include "DQM/EcalCommon/interface/MESetChannel.h"
#include "DQM/EcalCommon/interface/MESetDet0D.h"
#include "DQM/EcalCommon/interface/MESetDet1D.h"
#include "DQM/EcalCommon/interface/MESetDet2D.h"
#include "DQM/EcalCommon/interface/MESetEcal.h"
#include "DQM/EcalCommon/interface/MESetMulti.h"
#include "DQM/EcalCommon/interface/MESetNonObject.h"
#include "DQM/EcalCommon/interface/MESetProjection.h"
#include "DQM/EcalCommon/interface/MESetTrend.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace ecaldqm {
  using dqm::reco::MonitorElement;
  MESet *createMESet(edm::ParameterSet const &_MEParam) {
    std::string path(_MEParam.getUntrackedParameter<std::string>("path"));
    binning::ObjectType otype(binning::translateObjectType(_MEParam.getUntrackedParameter<std::string>("otype")));
    binning::BinningType btype(binning::translateBinningType(_MEParam.getUntrackedParameter<std::string>("btype")));
    MonitorElement::Kind kind(binning::translateKind(_MEParam.getUntrackedParameter<std::string>("kind")));

    binning::AxisSpecs xaxis, yaxis, zaxis;
    bool hasXaxis(_MEParam.existsAs<edm::ParameterSet>("xaxis", false));
    if (hasXaxis)
      xaxis = binning::formAxis(_MEParam.getUntrackedParameterSet("xaxis"));
    bool hasYaxis(_MEParam.existsAs<edm::ParameterSet>("yaxis", false));
    if (hasYaxis)
      yaxis = binning::formAxis(_MEParam.getUntrackedParameterSet("yaxis"));
    bool hasZaxis(_MEParam.existsAs<edm::ParameterSet>("zaxis", false));
    if (hasZaxis)
      zaxis = binning::formAxis(_MEParam.getUntrackedParameterSet("zaxis"));

    MESet *set(nullptr);

    if (btype == binning::kTrend) {
      MESetTrend *setTrend(new MESetTrend(path, otype, btype, kind, hasYaxis ? &yaxis : nullptr));
      if (_MEParam.existsAs<bool>("minutely", false) && _MEParam.getUntrackedParameter<bool>("minutely"))
        setTrend->setMinutely();
      if (_MEParam.existsAs<bool>("cumulative", false) && _MEParam.getUntrackedParameter<bool>("cumulative"))
        setTrend->setCumulative();
      if (_MEParam.existsAs<bool>("shiftAxis", false) && _MEParam.getUntrackedParameter<bool>("shiftAxis"))
        setTrend->setShiftAxis();
      set = setTrend;
    } else if (otype == binning::nObjType)
      set = new MESetNonObject(path,
                               otype,
                               btype,
                               kind,
                               hasXaxis ? &xaxis : nullptr,
                               hasYaxis ? &yaxis : nullptr,
                               hasZaxis ? &zaxis : nullptr);
    else if (otype == binning::kChannel)
// Class removed until concurrency issue is finalized
#if 0
      set = new MESetChannel(path, otype, btype, kind);
#else
      set = nullptr;
#endif
    else if (btype == binning::kProjEta || btype == binning::kProjPhi)
      set = new MESetProjection(path, otype, btype, kind, hasYaxis ? &yaxis : nullptr);
    else {
      unsigned logicalDimensions(-1);
      switch (kind) {
        case MonitorElement::Kind::REAL:
          logicalDimensions = 0;
          break;
        case MonitorElement::Kind::TH1F:
        case MonitorElement::Kind::TPROFILE:
          logicalDimensions = 1;
          break;
        case MonitorElement::Kind::TH2F:
        case MonitorElement::Kind::TPROFILE2D:
          logicalDimensions = 2;
          break;
        default:
          break;
      }

      // example case: Ecal/TriggerPrimitives/EmulMatching/TrigPrimTask matching
      // index
      if (logicalDimensions == 2 && hasYaxis && btype != binning::kUser)
        logicalDimensions = 1;

      if (logicalDimensions > 2 || (btype == binning::kReport && logicalDimensions != 0))
        throw cms::Exception("InvalidConfiguration") << "Cannot create MESet at " << path;

      if (btype == binning::kUser)
        set = new MESetEcal(path,
                            otype,
                            btype,
                            kind,
                            logicalDimensions,
                            hasXaxis ? &xaxis : nullptr,
                            hasYaxis ? &yaxis : nullptr,
                            hasZaxis ? &zaxis : nullptr);
      else if (logicalDimensions == 0)
        set = new MESetDet0D(path, otype, btype, kind);
      else if (logicalDimensions == 1)
        set = new MESetDet1D(path, otype, btype, kind, hasYaxis ? &yaxis : nullptr);
      else if (logicalDimensions == 2)
        set = new MESetDet2D(path, otype, btype, kind, hasZaxis ? &zaxis : nullptr);
    }

    if (_MEParam.existsAs<edm::ParameterSet>("multi", false)) {
      typedef std::vector<std::string> VString;

      edm::ParameterSet const &multiParams(_MEParam.getUntrackedParameterSet("multi"));
      VString replacementNames(multiParams.getParameterNames());
      if (replacementNames.empty())
        throw cms::Exception("InvalidConfiguration") << "0 multiplicity for MESet at " << path;

      MESetMulti::ReplCandidates candidates;
      for (unsigned iD(0); iD != replacementNames.size(); ++iD) {
        VString reps;
        if (multiParams.existsAs<VString>(replacementNames[iD], false))
          reps = multiParams.getUntrackedParameter<VString>(replacementNames[iD]);
        else if (multiParams.existsAs<std::vector<int>>(replacementNames[iD], false)) {
          std::vector<int> repInts(multiParams.getUntrackedParameter<std::vector<int>>(replacementNames[iD]));
          for (unsigned iR(0); iR != repInts.size(); ++iR)
            reps.push_back(std::to_string(repInts[iR]));
        }

        if (reps.empty())
          throw cms::Exception("InvalidConfiguration") << "0 multiplicity for MESet at " << path;

        candidates[replacementNames[iD]] = reps;
      }
      MESetMulti *multi(new MESetMulti(*set, candidates));
      delete set;
      set = multi;
    }

    if (!set)
      throw cms::Exception("InvalidConfiguration") << "MESet " << path << " could not be initialized";

    if (_MEParam.getUntrackedParameter<bool>("perLumi"))
      set->setLumiFlag();

    return set;
  }

  void fillMESetDescriptions(edm::ParameterSetDescription &_desc) {
    _desc.addUntracked<std::string>("path");
    _desc.addUntracked<std::string>("kind");
    _desc.addUntracked<std::string>("otype");
    _desc.addUntracked<std::string>("btype");
    _desc.addUntracked<std::string>("description");
    _desc.addUntracked<bool>("online", false);
    _desc.addUntracked<bool>("perLumi", false);
    _desc.addOptionalUntracked<bool>("minutely");
    _desc.addOptionalUntracked<bool>("cumulative");
    _desc.addOptionalUntracked<bool>("shiftAxis");

    edm::ParameterSetDescription axisParameters;
    binning::fillAxisDescriptions(axisParameters);
    _desc.addOptionalUntracked("xaxis", axisParameters);
    _desc.addOptionalUntracked("yaxis", axisParameters);
    _desc.addOptionalUntracked("zaxis", axisParameters);

    edm::ParameterSetDescription multiParameters;
    multiParameters.addWildcardUntracked<std::vector<std::string>>("*");
    multiParameters.addWildcardUntracked<std::vector<int>>("*");
    _desc.addOptionalUntracked("multi", multiParameters);
  }
}  // namespace ecaldqm
