#include "../interface/PresampleClient.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

#include <cmath>

namespace ecaldqm {

  PresampleClient::PresampleClient(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerClient(_workerParams, _commonParams, "PresampleClient"),
    minChannelEntries_(_workerParams.getUntrackedParameter<int>("minChannelEntries")),
    minTowerEntries_(_workerParams.getUntrackedParameter<int>("minTowerEntries")),
    expectedMean_(_workerParams.getUntrackedParameter<double>("expectedMean")),
    meanThreshold_(_workerParams.getUntrackedParameter<double>("meanThreshold")),
    rmsThreshold_(_workerParams.getUntrackedParameter<double>("rmsThreshold")),
    rmsThresholdHighEta_(_workerParams.getUntrackedParameter<double>("rmsThresholdHighEta"))
  {
  }

  void
  PresampleClient::bookMEs()
  {
    DQWorker::bookMEs();

    MEs_[kQuality]->resetAll(-1.);
    MEs_[kRMSMap]->resetAll(-1.);
    MEs_[kQualitySummary]->resetAll(-1.);
  }

  void
  PresampleClient::producePlots()
  {
    MEs_[kMean]->reset();
    MEs_[kMeanDCC]->reset();
    MEs_[kRMS]->reset();

    uint32_t mask(1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_MEAN_ERROR |
		  1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_RMS_ERROR);

    MESet::iterator qEnd(MEs_[kQuality]->end());

    MESet::iterator rItr(MEs_[kRMSMap]);
    MESet::const_iterator pItr(sources_[kPedestal]);
    for(MESet::iterator qItr(MEs_[kQuality]->beginChannel()); qItr != qEnd; qItr.toNextChannel()){

      pItr = qItr;
      rItr = qItr;

      DetId id(qItr->getId());

      double rmsThresh(rmsThreshold_);

      if(id.subdetId() == EcalEndcap){
        EEDetId eeid(id);
        if((eeid.ix() - 50) * (eeid.ix() - 50) + (eeid.iy() - 50) * (eeid.iy() - 50) < 400) rmsThresh = rmsThresholdHighEta_;
      }

      double entries(pItr->getBinEntries());

      if(entries < minChannelEntries_){
        qItr->setBinContent(maskQuality_(qItr, mask, 2));
        rItr->setBinContent(-1.);
        continue;
      }

      double mean(pItr->getBinContent());
      double rms(pItr->getBinError() * std::sqrt(entries));

      unsigned dccid(dccId(id));

      MEs_[kMean]->fill(dccid, mean);
      MEs_[kMeanDCC]->fill(dccid, mean);
      MEs_[kRMS]->fill(dccid, rms);
      rItr->setBinContent(rms);

      if(std::abs(mean - expectedMean_) > meanThreshold_ || rms > rmsThresh)
        qItr->setBinContent(maskQuality_(qItr, mask, 0));
      else
        qItr->setBinContent(maskQuality_(qItr, mask, 1));

    }

    towerAverage_(kQualitySummary, kQuality, 0.2);
  }

  /*static*/
  void
  PresampleClient::setMEOrdering(std::map<std::string, unsigned>& _nameToIndex)
  {
    _nameToIndex["Quality"] = kQuality;
    _nameToIndex["Mean"] = kMean;
    _nameToIndex["MeanDCC"] = kMeanDCC;
    _nameToIndex["RMS"] = kRMS;
    _nameToIndex["RMSMap"] = kRMSMap;
    _nameToIndex["QualitySummary"] = kQualitySummary;

    _nameToIndex["Pedestal"] = kPedestal;
  }

  DEFINE_ECALDQM_WORKER(PresampleClient);
}


