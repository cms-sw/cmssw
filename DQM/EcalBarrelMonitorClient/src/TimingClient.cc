#include "../interface/TimingClient.h"

#include "DQM/EcalBarrelMonitorTasks/interface/TimingTask.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include <cmath>

namespace ecaldqm {

  TimingClient::TimingClient(const edm::ParameterSet& _params, const edm::ParameterSet& _paths) :
    DQWorkerClient(_params, _paths, "TimingClient"),
    expectedMean_(0.),
    meanThreshold_(0.),
    rmsThreshold_(0.),
    minChannelEntries_(0),
    minTowerEntries_(0),
    tailPopulThreshold_(0.)
  {
    edm::ParameterSet const& taskParams(_params.getUntrackedParameterSet(name_));
    expectedMean_ = taskParams.getUntrackedParameter<double>("expectedMean");
    meanThreshold_ = taskParams.getUntrackedParameter<double>("meanThreshold");
    rmsThreshold_ = taskParams.getUntrackedParameter<double>("rmsThreshold");
    minChannelEntries_ = taskParams.getUntrackedParameter<int>("minChannelEntries");
    minTowerEntries_ = taskParams.getUntrackedParameter<int>("minTowerEntries");
    tailPopulThreshold_ = taskParams.getUntrackedParameter<double>("tailPopulThreshold");

    edm::ParameterSet const& sources(_params.getUntrackedParameterSet("sources"));
    source_(sTimeAllMap, "TimingTask", TimingTask::kTimeAllMap, sources);
    source_(sTimeMap, "TimingTask", TimingTask::kTimeMap, sources);
  }

  void
  TimingClient::bookMEs()
  {
    DQWorker::bookMEs();

    MEs_[kQuality]->resetAll(-1.);
    MEs_[kRMS]->resetAll(-1.);
    MEs_[kQualitySummary]->resetAll(-1.);
  }

  void
  TimingClient::producePlots()
  {
    using namespace std;

    MEs_[kMeanSM]->reset();
    MEs_[kMeanAll]->reset();
    MEs_[kRMS]->reset(-1.);
    MEs_[kRMSAll]->reset();
    MEs_[kProjEta]->reset();
    MEs_[kProjPhi]->reset();
    MEs_[kFwdBkwdDiff]->reset();
    MEs_[kFwdvBkwd]->reset();

    uint32_t mask(1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_WARNING);

    for(unsigned dccid(1); dccid <= 54; dccid++){

      for(unsigned tower(1); tower <= getNSuperCrystals(dccid); tower++){
	vector<DetId> ids(getElectronicsMap()->dccTowerConstituents(dccid, tower));

	if(ids.size() == 0) continue;

	// tower entries != sum(channel entries) because of the difference in timing cut at the source
	float summaryEntries(0.);
	if(dccid <= 9 || dccid >= 46){
	  vector<EcalScDetId> scids(getElectronicsMap()->getEcalScDetId(dccid, tower));
	  for(vector<EcalScDetId>::iterator scItr(scids.begin()); scItr != scids.end(); ++scItr)
	    summaryEntries += sources_[sTimeAllMap]->getBinEntries(*scItr);
	}
	else
	  summaryEntries = sources_[sTimeAllMap]->getBinEntries(ids[0]);

	float towerEntries(0.);
	float towerMean(0.);
	float towerMean2(0.);

	for(vector<DetId>::iterator idItr(ids.begin()); idItr != ids.end(); ++idItr){
	  float entries(sources_[sTimeMap]->getBinEntries(*idItr));
	  float mean(sources_[sTimeMap]->getBinContent(*idItr));
	  float rms(sources_[sTimeMap]->getBinError(*idItr) * sqrt(entries));

	  towerEntries += entries;
	  towerMean += mean;
	  towerMean2 += mean * mean;

	  if(entries < minChannelEntries_){
	    fillQuality_(kQuality, *idItr, mask, 2.);
	    continue;
	  }

	  MEs_[kMeanSM]->fill(*idItr, mean);
	  MEs_[kMeanAll]->fill(*idItr, mean);
	  MEs_[kProjEta]->fill(*idItr, mean);
	  MEs_[kProjPhi]->fill(*idItr, mean);
	  MEs_[kRMS]->fill(*idItr, rms);
	  MEs_[kRMSAll]->fill(*idItr, rms);

	  if(dccid <= 27){
	    DetId posId(0);
	    if(idItr->subdetId() == EcalEndcap){
	      posId = EEDetId::switchZSide(*idItr);
	    }
	    else{
	      posId = EBDetId::switchZSide(*idItr);
	    }
	    float posTime(sources_[sTimeMap]->getBinContent(posId));
	    MEs_[kFwdBkwdDiff]->fill(*idItr, posTime - mean);
	    MEs_[kFwdvBkwd]->fill(*idItr, mean, posTime);
	  }

	  float quality(abs(mean - expectedMean_) > meanThreshold_ || rms > rmsThreshold_ ? 0. : 1.);
	  fillQuality_(kQuality, *idItr, mask, quality);
	}

	float quality(1.);
	if(towerEntries > minTowerEntries_){
	  if(summaryEntries < towerEntries * (1. - tailPopulThreshold_)) // large timing deviation
	    quality = 0.;

	  towerMean /= ids.size();
	  towerMean2 /= ids.size();

	  float towerRMS(0.);
	  float variance(towerMean2 - towerMean * towerMean);
	  if(variance > 0.) towerRMS = sqrt(variance);

	  if(abs(towerMean - expectedMean_) > meanThreshold_ || towerRMS > rmsThreshold_)
	    quality = 0.;
	}
	else
	  quality = 2.;

	if(dccid <= 9 || dccid >= 46){
	  vector<EcalScDetId> scs(getElectronicsMap()->getEcalScDetId(dccid, tower));
	  for(vector<EcalScDetId>::iterator scItr(scs.begin()); scItr != scs.end(); ++scItr)
	    fillQuality_(kQualitySummary, *scItr, mask, quality);
	}
	else
	  fillQuality_(kQualitySummary, ids[0], mask, quality);
      }
    }
  }

  /*static*/
  void
  TimingClient::setMEData(std::vector<MEData>& _data)
  {
    BinService::AxisSpecs axis;

    _data[kQuality] = MEData("Quality", BinService::kSM, BinService::kCrystal, MonitorElement::DQM_KIND_TH2F);

    axis.nbins = 100;
    axis.low = -25.;
    axis.high = 25.;
    _data[kMeanSM] = MEData("MeanSM", BinService::kSM, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &axis);
    _data[kMeanAll] = MEData("MeanAll", BinService::kEcal2P, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &axis);

    axis.nbins = 50;
    _data[kFwdvBkwd] = MEData("FwdvBkwd", BinService::kEcal2P, BinService::kUser, MonitorElement::DQM_KIND_TH2F, &axis, &axis);

    axis.low = -5.;
    axis.high = 5.;
    _data[kFwdBkwdDiff] = MEData("FwdBkwdDiff", BinService::kEcal2P, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &axis);

    _data[kRMS] = MEData("RMS", BinService::kSM, BinService::kCrystal, MonitorElement::DQM_KIND_TH2F);

    axis.nbins = 100;
    axis.low = 0.;
    axis.high = 10.;
    _data[kRMSAll] = MEData("RMSAll", BinService::kEcal2P, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &axis);
    _data[kProjEta] = MEData("Projection", BinService::kEcal3P, BinService::kProjEta, MonitorElement::DQM_KIND_TPROFILE);
    _data[kProjPhi] = MEData("Projection", BinService::kEcal3P, BinService::kProjPhi, MonitorElement::DQM_KIND_TPROFILE);
    _data[kQualitySummary] = MEData("QualitySummary", BinService::kEcal2P, BinService::kSuperCrystal, MonitorElement::DQM_KIND_TH2F);
  }

  DEFINE_ECALDQM_WORKER(TimingClient);
}
