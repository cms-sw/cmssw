#include "../interface/PresampleClient.h"

#include "DQM/EcalBarrelMonitorTasks/interface/PresampleTask.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include <cmath>

namespace ecaldqm {

  PresampleClient::PresampleClient(const edm::ParameterSet& _params, const edm::ParameterSet& _paths) :
    DQWorkerClient(_params, _paths, "PresampleClient"),
    minChannelEntries_(0),
    minTowerEntries_(0),
    expectedMean_(0.),
    meanThreshold_(0.),
    rmsThreshold_(0.),
    rmsThresholdHighEta_(0.),
    noisyFracThreshold_(0.)
  {
    edm::ParameterSet const& taskParams(_params.getUntrackedParameterSet(name_));
    minChannelEntries_ = taskParams.getUntrackedParameter<int>("minChannelEntries");
    minTowerEntries_ = taskParams.getUntrackedParameter<int>("minTowerEntries");
    expectedMean_ = taskParams.getUntrackedParameter<double>("expectedMean");
    meanThreshold_ = taskParams.getUntrackedParameter<double>("meanThreshold");
    rmsThreshold_ = taskParams.getUntrackedParameter<double>("rmsThreshold");
    rmsThresholdHighEta_ = taskParams.getUntrackedParameter<double>("rmsThresholdHighEta");
    noisyFracThreshold_ = taskParams.getUntrackedParameter<double>("noisyFracThreshold");

    edm::ParameterSet const& sources(_params.getUntrackedParameterSet("sources"));
    source_(sPedestal, "PresampleTask", PresampleTask::kPedestal, sources);
  }

  void
  PresampleClient::bookMEs()
  {
    DQWorker::bookMEs();

    MEs_[kQuality]->resetAll(-1.);
    MEs_[kRMSMap]->resetAll(-1.);
    MEs_[kRMSMapSummary]->resetAll(-1.);
    MEs_[kQualitySummary]->resetAll(-1.);
  }

  void
  PresampleClient::producePlots()
  {
    MEs_[kMean]->reset();
    MEs_[kMeanDCC]->reset();
    MEs_[kRMS]->reset();
    MEs_[kRMSMap]->reset(-1.);
    MEs_[kRMSMapSummary]->reset(-1.);

    uint32_t mask(1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_MEAN_ERROR |
		  1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_RMS_ERROR);

    for(unsigned dccid(1); dccid <= 54; dccid++){

      for(unsigned tower(1); tower <= getNSuperCrystals(dccid); tower++){
	std::vector<DetId> ids(getElectronicsMap()->dccTowerConstituents(dccid, tower));

	if(ids.size() == 0) continue;

	unsigned iSM(dccid - 1);
	float rmsThresh(rmsThreshold_);
	if(iSM <= kEEmHigh || iSM >= kEEpLow || tower > 48) rmsThresh = rmsThresholdHighEta_;

	float nNoisy(0.);
	float towerEntries(0.);
	float towerMean(0.);
	float towerRMS(0.);

	for(std::vector<DetId>::iterator idItr(ids.begin()); idItr != ids.end(); ++idItr){
	  float entries(sources_[sPedestal]->getBinEntries(*idItr));
	  float mean(sources_[sPedestal]->getBinContent(*idItr));
	  float rms(sources_[sPedestal]->getBinError(*idItr) * std::sqrt(entries));
	  towerEntries += entries;
	  towerMean += mean * entries;
	  towerRMS += (rms * rms + mean * mean) * entries;

	  if(entries < minChannelEntries_){
	    fillQuality_(kQuality, *idItr, mask, 2.);
	    continue;
	  }

	  MEs_[kMean]->fill(*idItr, mean);
	  MEs_[kMeanDCC]->fill(*idItr, mean);
	  MEs_[kRMS]->fill(*idItr, rms);
	  MEs_[kRMSMap]->fill(*idItr, rms);

	  if(std::abs(mean - expectedMean_) > meanThreshold_ || rms > rmsThresh){
	    fillQuality_(kQuality, *idItr, mask, 0.);
	    nNoisy += 1.;
	  }
	  else
	    fillQuality_(kQuality, *idItr, mask, 1.);
	}

	towerMean /= towerEntries;
	towerRMS = std::sqrt(towerRMS / towerEntries - towerMean * towerMean);

	float quality(-1.);

	if(towerEntries > minTowerEntries_)
	  quality = nNoisy / ids.size() > noisyFracThreshold_ ? 0. : 1.;
	else
	  quality = 2.;

	if(dccid <= 9 || dccid >= 46){
	  std::vector<EcalScDetId> scs(getElectronicsMap()->getEcalScDetId(dccid, tower));
	  for(std::vector<EcalScDetId>::iterator scItr(scs.begin()); scItr != scs.end(); ++scItr){
	    fillQuality_(kQualitySummary, *scItr, mask, quality);
	    MEs_[kRMSMapSummary]->setBinContent(*scItr, towerRMS);
	  }
	}
	else{
	  fillQuality_(kQualitySummary, ids[0], mask, quality);
	  MEs_[kRMSMapSummary]->setBinContent(ids[0], towerRMS);
	}
      }
    }
  }

  /*static*/
  void
  PresampleClient::setMEData(std::vector<MEData>& _data)
  {
    BinService::AxisSpecs axis;

    _data[kQuality] = MEData("Quality", BinService::kSM, BinService::kCrystal, MonitorElement::DQM_KIND_TH2F);

    axis.nbins = 120;
    axis.low = 170.;
    axis.high = 230.;
    _data[kMean] = MEData("Mean", BinService::kSM, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &axis);
    _data[kMeanDCC] = MEData("MeanDCC", BinService::kEcal2P, BinService::kDCC, MonitorElement::DQM_KIND_TPROFILE, 0, &axis);

    axis.nbins = 100;
    axis.low = 0.;
    axis.high = 10.;
    _data[kRMS] = MEData("RMS", BinService::kSM, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &axis);
    _data[kRMSMap] = MEData("RMSMap", BinService::kSM, BinService::kCrystal, MonitorElement::DQM_KIND_TH2F, 0, 0, &axis);
    _data[kRMSMapSummary] = MEData("RMSMap", BinService::kEcal2P, BinService::kSuperCrystal, MonitorElement::DQM_KIND_TH2F, 0, 0, &axis);

    _data[kQualitySummary] = MEData("QualitySummary", BinService::kEcal2P, BinService::kSuperCrystal, MonitorElement::DQM_KIND_TH2F);
  }

  DEFINE_ECALDQM_WORKER(PresampleClient);
}

