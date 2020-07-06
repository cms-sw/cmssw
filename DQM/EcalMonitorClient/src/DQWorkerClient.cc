#include "DQM/EcalMonitorClient/interface/DQWorkerClient.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "DQM/EcalCommon/interface/StatusManager.h"
#include "DQM/EcalCommon/interface/MESetChannel.h"
#include "DQM/EcalCommon/interface/MESetMulti.h"
#include "DQM/EcalCommon/interface/MESetUtils.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <sstream>

namespace ecaldqm {
  DQWorkerClient::DQWorkerClient()
      : DQWorker(), sources_(), qualitySummaries_(), hasLumiPlots_(false), statusManager_(nullptr) {}

  /*static*/
  void DQWorkerClient::fillDescriptions(edm::ParameterSetDescription& _desc) {
    DQWorker::fillDescriptions(_desc);
    _desc.addWildcardUntracked<std::vector<std::string> >("*");

    edm::ParameterSetDescription sourceParameters;
    edm::ParameterSetDescription sourceNodeParameters;
    fillMESetDescriptions(sourceNodeParameters);
    sourceParameters.addNode(
        edm::ParameterWildcard<edm::ParameterSetDescription>("*", edm::RequireZeroOrMore, false, sourceNodeParameters));
    _desc.addUntracked("sources", sourceParameters);
  }

  void DQWorkerClient::setME(edm::ParameterSet const& _ps) {
    DQWorker::setME(_ps);

    // Flags the Client ME to run as lumibased:
    // In offline mode will save the ME client at the end of the LS
    // See: EcalDQMonitorClient::dqmEndLuminosityBlock
    for (auto& mItr : MEs_) {
      if (mItr.second->getLumiFlag()) {
        hasLumiPlots_ = true;
        break;
      }
    }
  }

  void DQWorkerClient::setSource(edm::ParameterSet const& _params) {
    std::vector<std::string> const& sourceNames(_params.getParameterNames());

    for (unsigned iS(0); iS < sourceNames.size(); iS++) {
      std::string name(sourceNames[iS]);
      edm::ParameterSet const& params(_params.getUntrackedParameterSet(name));

      if (onlineMode_ && params.getUntrackedParameter<bool>("online"))
        continue;

      sources_.insert(name, createMESet(params));
    }

    if (verbosity_ > 1) {
      std::stringstream ss;
      ss << name_ << ": Using ";
      for (MESetCollection::const_iterator sItr(sources_.begin()); sItr != sources_.end(); ++sItr)
        ss << sItr->first << " ";
      ss << "as sources";
      edm::LogInfo("EcalDQM") << ss.str();
    }
  }

  void DQWorkerClient::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {
// MESetChannel class removed until concurrency issue is finalized
#if 0
    for(MESetCollection::iterator sItr(sources_.begin()); sItr != sources_.end(); ++sItr){
      if(!sItr->second->getLumiFlag()) continue;
      MESetChannel const* channel(dynamic_cast<MESetChannel const*>(sItr->second));
      if(channel) channel->checkDirectory();
    }
#endif
  }

  void DQWorkerClient::bookMEs(DQMStore::IBooker& _ibooker) {
    DQWorker::bookMEs(_ibooker);
    resetMEs();
  }

  void DQWorkerClient::releaseMEs() {
    DQWorker::releaseMEs();
    releaseSource();
  }

  void DQWorkerClient::releaseSource() {
    for (MESetCollection::iterator sItr(sources_.begin()); sItr != sources_.end(); ++sItr)
      sItr->second->clear();
  }

  bool DQWorkerClient::retrieveSource(DQMStore::IGetter& _igetter, ProcessType _type) {
    std::string failedPath;
    for (MESetCollection::iterator sItr(sources_.begin()); sItr != sources_.end(); ++sItr) {
      if (!onlineMode_ && _type == kLumi && !sItr->second->getLumiFlag())
        continue;
      if (verbosity_ > 1)
        edm::LogInfo("EcalDQM") << name_ << ": Retrieving source " << sItr->first;
      if (!sItr->second->retrieve(_igetter, &failedPath)) {
        if (verbosity_ > 1)
          edm::LogWarning("EcalDQM") << name_ << ": Could not find source " << sItr->first << "@" << failedPath;
        return false;
      }
    }

    return true;
  }

  void DQWorkerClient::resetMEs() {
    for (auto& mItr : MEs_) {
      MESet* meset(mItr.second.get());

      // Protects Trend-type Client MEs from being reset at the end of the LS
      // See: EcalDQMonitorClient::runWorkers
      if (meset->getBinType() == ecaldqm::binning::kTrend)
        continue;

      if (qualitySummaries_.find(mItr.first) != qualitySummaries_.end()) {
        MESetMulti* multi(dynamic_cast<MESetMulti*>(meset));
        if (multi) {
          for (unsigned iS(0); iS < multi->getMultiplicity(); ++iS) {
            multi->use(iS);
            if (multi->getKind() == MonitorElement::Kind::TH2F) {
              multi->resetAll(-1.);
              multi->reset(kUnknown);
            } else
              multi->reset(-1.);
          }
        } else {
          if (meset->getKind() == MonitorElement::Kind::TH2F) {
            meset->resetAll(-1.);
            meset->reset(kUnknown);
          } else
            meset->reset(-1.);
        }
      } else
        meset->reset();
    }
  }

  void DQWorkerClient::resetPerLumi() {
    for (auto const& meset : MEs_) {
      int i = 0;
      while (auto me = meset.second->getME(i)) {
        if (me->getLumiFlag()) {
          // reset per-lumi histograms in offline harvesting so that they only show
          // data of the current lumisection.
          me->Reset();
        }
        i++;
      }
    }
  }

  void DQWorkerClient::towerAverage_(MESet& _target, MESet const& _source, float _threshold) {
    bool isQuality(_threshold > 0.);

    MESet::iterator tEnd(_target.end());
    for (MESet::iterator tItr(_target.beginChannel()); tItr != tEnd; tItr.toNextChannel()) {
      DetId towerId(tItr->getId());

      std::vector<DetId> cryIds;
      if (towerId.subdetId() == EcalTriggerTower)
        cryIds = getTrigTowerMap()->constituentsOf(EcalTrigTowerDetId(towerId));
      else {
        cryIds = scConstituents(EcalScDetId(towerId));
      }

      if (cryIds.empty())
        return;

      float mean(0.);
      float nValid(0.);
      bool masked(false);
      for (unsigned iId(0); iId < cryIds.size(); ++iId) {
        float content(_source.getBinContent(cryIds[iId]));
        if (isQuality) {
          if (content < 0. || content == 2.)
            continue;
          if (content == 5.)
            masked = true;
          else {
            nValid += 1;
            if (content > 2.) {
              masked = true;
              mean += content - 3.;
            } else
              mean += content;
          }
        } else {
          mean += content;
          nValid += 1.;
        }
      }

      if (isQuality) {
        if (nValid < 1.)
          tItr->setBinContent(masked ? 5. : 2.);
        else {
          mean /= nValid;
          if (mean < _threshold)
            tItr->setBinContent(masked ? 3. : 0.);
          else
            tItr->setBinContent(masked ? 4. : 1.);
        }
      } else
        tItr->setBinContent(nValid < 1. ? 0. : mean / nValid);
    }
  }

}  // namespace ecaldqm
