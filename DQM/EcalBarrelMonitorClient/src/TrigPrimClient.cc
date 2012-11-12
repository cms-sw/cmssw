#include "../interface/TrigPrimClient.h"
#include "../interface/EcalDQMClientUtils.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

#include <cmath>

namespace ecaldqm {

  TrigPrimClient::TrigPrimClient(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerClient(_workerParams, _commonParams, "TrigPrimClient"),
    minEntries_(_workerParams.getUntrackedParameter<int>("minEntries")),
    errorFractionThreshold_(_workerParams.getUntrackedParameter<double>("errorFractionThreshold"))
  {
    qualitySummaries_.insert("EmulQualitySummary");
  }

  void
  TrigPrimClient::producePlots()
  {
    MESet* meTimingSummary(MEs_["TimingSummary"]);
    MESet* meNonSingleSummary(MEs_["NonSingleSummary"]);
    MESet* meEmulQualitySummary(MEs_["EmulQualitySummary"]);

    MESet const* sEtEmulError(sources_["EtEmulError"]);
    MESet const* sMatchedIndex(sources_["MatchedIndex"]);

    //    meTiming->reset();
    meTimingSummary->reset();
    meNonSingleSummary->reset();

    uint32_t mask(1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_WARNING);

    for(unsigned iTT(0); iTT < EcalTrigTowerDetId::kSizeForDenseIndexing; iTT++){
      EcalTrigTowerDetId ttid(EcalTrigTowerDetId::detIdFromDenseIndex(iTT));

      bool doMask(applyMask(meEmulQualitySummary->getBinType(), ttid, mask));

      float towerEntries(0.);
      float tMax(0.5);
      float nMax(0.);
      for(int iBin(0); iBin < 6; iBin++){
	float entries(sMatchedIndex->getBinContent(ttid, iBin + 1));
	towerEntries += entries;

	if(entries > nMax){
	  nMax = entries;
	  tMax = iBin == 0 ? -0.5 : iBin + 0.5; // historical reasons.. much clearer to say "no entry = -0.5"
	}
      }

      meTimingSummary->setBinContent(ttid, tMax);

      if(towerEntries < minEntries_){
	meEmulQualitySummary->setBinContent(ttid, doMask ? kMUnknown : kUnknown);
	continue;
      }

      float nonsingleFraction(1. - nMax / towerEntries);

      if(nonsingleFraction > 0.)
	meNonSingleSummary->setBinContent(ttid, nonsingleFraction);

      if(sEtEmulError->getBinContent(ttid) / towerEntries > errorFractionThreshold_)
        meEmulQualitySummary->setBinContent(ttid, doMask ? kMBad : kBad);
      else
        meEmulQualitySummary->setBinContent(ttid, doMask ? kMGood : kGood);
    }
  }

  DEFINE_ECALDQM_WORKER(TrigPrimClient);
}
