#include "../interface/TrigPrimClient.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <cmath>

namespace ecaldqm
{
  TrigPrimClient::TrigPrimClient() :
    DQWorkerClient(),
    minEntries_(0),
    errorFractionThreshold_(0.)
  {
    qualitySummaries_.insert("EmulQualitySummary");
  }
  
  void
  TrigPrimClient::setParams(edm::ParameterSet const& _params)
  {
    minEntries_ = _params.getUntrackedParameter<int>("minEntries");
    errorFractionThreshold_ = _params.getUntrackedParameter<double>("errorFractionThreshold");
  }

  void
  TrigPrimClient::producePlots(ProcessType)
  {
    MESet& meTimingSummary(MEs_.at("TimingSummary"));
    MESet& meNonSingleSummary(MEs_.at("NonSingleSummary"));
    MESet& meEmulQualitySummary(MEs_.at("EmulQualitySummary"));

    MESet const& sEtEmulError(sources_.at("EtEmulError"));
    MESet const& sMatchedIndex(sources_.at("MatchedIndex"));

    uint32_t mask(1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_WARNING);

    for(unsigned iTT(0); iTT < EcalTrigTowerDetId::kSizeForDenseIndexing; iTT++){
      EcalTrigTowerDetId ttid(EcalTrigTowerDetId::detIdFromDenseIndex(iTT));

      bool doMask(meEmulQualitySummary.maskMatches(ttid, mask, statusManager_));

      float towerEntries(0.);
      float tMax(0.5);
      float nMax(0.);
      for(int iBin(0); iBin < 6; iBin++){
	float entries(sMatchedIndex.getBinContent(ttid, iBin + 1));
	towerEntries += entries;

	if(entries > nMax){
	  nMax = entries;
	  tMax = iBin == 0 ? -0.5 : iBin + 0.5; // historical reasons.. much clearer to say "no entry = -0.5"
	}
      }

      meTimingSummary.setBinContent(ttid, tMax);

      if(towerEntries < minEntries_){
	meEmulQualitySummary.setBinContent(ttid, doMask ? kMUnknown : kUnknown);
	continue;
      }

      float nonsingleFraction(1. - nMax / towerEntries);

      if(nonsingleFraction > 0.)
	meNonSingleSummary.setBinContent(ttid, nonsingleFraction);

      if(sEtEmulError.getBinContent(ttid) / towerEntries > errorFractionThreshold_)
        meEmulQualitySummary.setBinContent(ttid, doMask ? kMBad : kBad);
      else
        meEmulQualitySummary.setBinContent(ttid, doMask ? kMGood : kGood);
    }
  }

  DEFINE_ECALDQM_WORKER(TrigPrimClient);
}
