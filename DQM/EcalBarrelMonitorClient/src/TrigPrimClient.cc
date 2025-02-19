#include "../interface/TrigPrimClient.h"

#include "DQM/EcalBarrelMonitorTasks/interface/TrigPrimTask.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include <cmath>

namespace ecaldqm {

  TrigPrimClient::TrigPrimClient(const edm::ParameterSet& _params, const edm::ParameterSet& _paths) :
    DQWorkerClient(_params, _paths, "TrigPrimClient")
  {
    edm::ParameterSet const& sources(_params.getUntrackedParameterSet("sources"));
    source_(sEtRealMap, "TrigPrimTask", TrigPrimTask::kEtRealMap, sources);
    source_(sEtEmulError, "TrigPrimTask", TrigPrimTask::kEtEmulError, sources);
    source_(sTimingError, "TrigPrimTask", TrigPrimTask::kTimingError, sources);
    source_(sMatchedIndex, "TrigPrimTask", TrigPrimTask::kMatchedIndex, sources);
  }

  void
  TrigPrimClient::bookMEs()
  {
    DQWorker::bookMEs();

    MEs_[kEmulQualitySummary]->resetAll(-1.);
  }

  void
  TrigPrimClient::producePlots()
  {
    //    MEs_[kTiming]->reset();
    MEs_[kTimingSummary]->reset();
    MEs_[kNonSingleSummary]->reset();

    uint32_t mask(1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_WARNING);

    for(unsigned iTT(0); iTT < EcalTrigTowerDetId::kSizeForDenseIndexing; iTT++){
      EcalTrigTowerDetId ttid(EcalTrigTowerDetId::detIdFromDenseIndex(iTT));

      float towerEntries(0.);
      float tMax(0.5);
      float nMax(0.);
      for(int iBin(0); iBin < 6; iBin++){
	float entries(sources_[sMatchedIndex]->getBinEntries(ttid, iBin + 1));
	towerEntries += entries;

	if(entries > nMax){
	  nMax = entries;
	  tMax = iBin == 0 ? -0.5 : iBin + 0.5; // historical reasons.. much clearer to say "no entry = -0.5"
	}
      }

      //      MEs_[kTiming]->setBinContent(ttid, tMax);
      MEs_[kTimingSummary]->setBinContent(ttid, tMax);

      if(towerEntries < 1.){
	fillQuality_(kEmulQualitySummary, ttid, mask, 2.);
	continue;
      }

      float nonsingleFraction(1. - nMax / towerEntries);

      if(nonsingleFraction > 0.)
	MEs_[kNonSingleSummary]->setBinContent(ttid, nonsingleFraction);

      float quality(sources_[sEtEmulError]->getBinContent(ttid) > 0. || sources_[sTimingError]->getBinContent(ttid) > 0. ? 0. : 1.);
      fillQuality_(kEmulQualitySummary, ttid, mask, quality);
    }
  }

  /*static*/
  void
  TrigPrimClient::setMEData(std::vector<MEData>& _data)
  {
    _data[kTimingSummary] = MEData("TimingSummary", BinService::kEcal2P, BinService::kTriggerTower, MonitorElement::DQM_KIND_TH2F);
    _data[kNonSingleSummary] = MEData("NonSingleSummary", BinService::kEcal2P, BinService::kTriggerTower, MonitorElement::DQM_KIND_TH2F);
    _data[kEmulQualitySummary] = MEData("EmulQualitySummary", BinService::kEcal2P, BinService::kTriggerTower, MonitorElement::DQM_KIND_TH2F);
  }

  DEFINE_ECALDQM_WORKER(TrigPrimClient);
}

