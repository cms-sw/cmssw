#include "../interface/TrigPrimClient.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

#include <cmath>

namespace ecaldqm {

  TrigPrimClient::TrigPrimClient(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerClient(_workerParams, _commonParams, "TrigPrimClient")
  {
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
	float entries(sources_[kMatchedIndex]->getBinContent(ttid, iBin + 1));
	towerEntries += entries;

	if(entries > nMax){
	  nMax = entries;
	  tMax = iBin == 0 ? -0.5 : iBin + 0.5; // historical reasons.. much clearer to say "no entry = -0.5"
	}
      }

      MEs_[kTimingSummary]->setBinContent(ttid, tMax);

      if(towerEntries < 1.){
	MEs_[kEmulQualitySummary]->setBinContent(ttid, maskQuality_(kEmulQualitySummary, ttid, mask, 2));
	continue;
      }

      float nonsingleFraction(1. - nMax / towerEntries);

      if(nonsingleFraction > 0.)
	MEs_[kNonSingleSummary]->setBinContent(ttid, nonsingleFraction);

      int quality(sources_[kEtEmulError]->getBinContent(ttid) > 0. || sources_[kTimingError]->getBinContent(ttid) > 0. ? 0 : 1);
      MEs_[kEmulQualitySummary]->setBinContent(ttid, maskQuality_(kEmulQualitySummary, ttid, mask, quality));
    }
  }

  /*static*/
  void
  TrigPrimClient::setMEOrdering(std::map<std::string, unsigned>& _nameToIndex)
  {
    _nameToIndex["TimingSummary"] = kTimingSummary;
    _nameToIndex["NonSingleSummary"] = kNonSingleSummary;
    _nameToIndex["EmulQualitySummary"] = kEmulQualitySummary;

    _nameToIndex["EtRealMap"] = kEtRealMap;
    _nameToIndex["EtEmulError"] = kEtEmulError;
    _nameToIndex["TimingError"] = kTimingError;
    _nameToIndex["MatchedIndex"] = kMatchedIndex;
  }

  DEFINE_ECALDQM_WORKER(TrigPrimClient);
}

