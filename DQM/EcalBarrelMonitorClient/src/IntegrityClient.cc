#include "../interface/IntegrityClient.h"

#include "DQM/EcalBarrelMonitorTasks/interface/OccupancyTask.h"
#include "DQM/EcalBarrelMonitorTasks/interface/IntegrityTask.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

namespace ecaldqm {

  IntegrityClient::IntegrityClient(const edm::ParameterSet& _params, const edm::ParameterSet& _paths) :
    DQWorkerClient(_params, _paths, "IntegrityClient"),
    errFractionThreshold_(0.)
  {
    edm::ParameterSet const& taskParams(_params.getUntrackedParameterSet(name_));
    errFractionThreshold_ = taskParams.getUntrackedParameter<double>("errFractionThreshold");

    edm::ParameterSet const& sources(_params.getUntrackedParameterSet("sources"));
    source_(sOccupancy, "OccupancyTask", OccupancyTask::kDigi, sources);
    source_(sGain, "IntegrityTask", IntegrityTask::kGain, sources);
    source_(sChId, "IntegrityTask", IntegrityTask::kChId, sources);
    source_(sGainSwitch, "IntegrityTask", IntegrityTask::kGainSwitch, sources);
    source_(sTowerId, "IntegrityTask", IntegrityTask::kTowerId, sources);
    source_(sBlockSize, "IntegrityTask", IntegrityTask::kBlockSize, sources);
  }

  void
  IntegrityClient::bookMEs()
  {
    DQWorker::bookMEs();

    MEs_[kQuality]->resetAll(-1.);
    MEs_[kQualitySummary]->resetAll(-1.);
  }

  void
  IntegrityClient::producePlots()
  {
    uint32_t mask(1 << EcalDQMStatusHelper::CH_ID_ERROR |
		  1 << EcalDQMStatusHelper::CH_GAIN_ZERO_ERROR |
		  1 << EcalDQMStatusHelper::CH_GAIN_SWITCH_ERROR |
		  1 << EcalDQMStatusHelper::TT_ID_ERROR |
		  1 << EcalDQMStatusHelper::TT_SIZE_ERROR);

    for(unsigned dccid(1); dccid <= 54; dccid++){
      for(unsigned tower(1); tower <= getNSuperCrystals(dccid); tower++){
	std::vector<DetId> ids(getElectronicsMap()->dccTowerConstituents(dccid, tower));

	if(ids.size() == 0) continue;

	float towerEntries(0.);
	bool towerGood(true);

	for(std::vector<DetId>::iterator idItr(ids.begin()); idItr != ids.end(); ++idItr){
	  float entries(sources_[sOccupancy]->getBinContent(*idItr));
	  towerEntries += entries;

	  float gain(sources_[sGain]->getBinContent(*idItr));
	  float chid(sources_[sChId]->getBinContent(*idItr));
	  float gainswitch(sources_[sGainSwitch]->getBinContent(*idItr));

	  if(entries + gain + chid + gainswitch < 1.){
	    fillQuality_(kQuality, *idItr, mask, 2.);
	    continue;
	  }

	  float chErr((gain + chid + gainswitch) / (entries + gain + chid + gainswitch));

	  if(chErr > errFractionThreshold_){
	    fillQuality_(kQuality, *idItr, mask, 0.);
	    towerGood = false;
	  }
	  else
	    fillQuality_(kQuality, *idItr, mask, 1.);
	}

	float towerid(sources_[sTowerId]->getBinContent(ids[0]));
	float blocksize(sources_[sBlockSize]->getBinContent(ids[0]));

	float quality(-1.);

	if(towerEntries + towerid + blocksize > 1.){
	  float towerErr((towerid + blocksize) / (towerEntries + towerid + blocksize));
	  if(towerErr > errFractionThreshold_) towerGood = false;

	  quality = towerGood ? 1. : 0.;
	}
	else{
	  quality = 2.;
	}

	if(dccid <= 9 || dccid >= 46){
	  std::vector<EcalScDetId> scs(getElectronicsMap()->getEcalScDetId(dccid, tower));
	  for(std::vector<EcalScDetId>::iterator scItr(scs.begin()); scItr != scs.end(); ++scItr)
	    fillQuality_(kQualitySummary, *scItr, mask, quality);
	}
	else
	  fillQuality_(kQualitySummary, ids[0], mask, quality);
      }
    }
  }

  /*static*/
  void
  IntegrityClient::setMEData(std::vector<MEData>& _data)
  {
    _data[kQuality] = MEData("Quality", BinService::kSM, BinService::kCrystal, MonitorElement::DQM_KIND_TH2F);
    _data[kQualitySummary] = MEData("QualitySummary", BinService::kEcal2P, BinService::kSuperCrystal, MonitorElement::DQM_KIND_TH2F);
  }

  DEFINE_ECALDQM_WORKER(IntegrityClient);
}

