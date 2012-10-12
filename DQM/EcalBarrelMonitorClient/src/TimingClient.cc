#include "../interface/TimingClient.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

#include <cmath>

namespace ecaldqm {

  TimingClient::TimingClient(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerClient(_workerParams, _commonParams, "TimingClient"),
    toleranceMean_(_workerParams.getUntrackedParameter<double>("toleranceMean")),
    toleranceMeanFwd_(_workerParams.getUntrackedParameter<double>("toleranceMeanFwd")),
    toleranceRMS_(_workerParams.getUntrackedParameter<double>("toleranceRMS")),
    toleranceRMSFwd_(_workerParams.getUntrackedParameter<double>("toleranceRMSFwd")),
    minChannelEntries_(_workerParams.getUntrackedParameter<int>("minChannelEntries")),
    minChannelEntriesFwd_(_workerParams.getUntrackedParameter<int>("minChannelEntriesFwd")),
    minTowerEntries_(_workerParams.getUntrackedParameter<int>("minTowerEntries")),
    minTowerEntriesFwd_(_workerParams.getUntrackedParameter<int>("minChannelEntriesFwd")),
    tailPopulThreshold_(_workerParams.getUntrackedParameter<double>("tailPopulThreshold"))
  {
    qualitySummaries_.insert(kQuality);
    qualitySummaries_.insert(kQualitySummary);
  }

  void
  TimingClient::producePlots()
  {
    using namespace std;

    MEs_[kMeanSM]->reset();
    MEs_[kMeanAll]->reset();
    MEs_[kRMSAll]->reset();
    MEs_[kProjEta]->reset();
    MEs_[kProjPhi]->reset();
    MEs_[kFwdBkwdDiff]->reset();
    MEs_[kFwdvBkwd]->reset();

    uint32_t mask(1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_WARNING);

    MESet::iterator qEnd(MEs_[kQuality]->end());

    MESet::iterator rItr(MEs_[kRMSMap]);
    MESet::const_iterator tItr(sources_[kTimeMap]);

    for(MESet::iterator qItr(MEs_[kQuality]->beginChannel()); qItr != qEnd; qItr.toNextChannel()){

      tItr = qItr;
      rItr = qItr;

      DetId id(qItr->getId());

      int minChannelEntries(minChannelEntries_);
      float meanThresh(toleranceMean_);
      float rmsThresh(toleranceRMS_);

      if(isForward(id)){
        minChannelEntries = minChannelEntriesFwd_;
        meanThresh = toleranceMeanFwd_;
        rmsThresh = toleranceRMSFwd_;
      }

      bool doMask(applyMask_(kQuality, id, mask));

      float entries(tItr->getBinEntries());

      if(entries < minChannelEntries){
        qItr->setBinContent(doMask ? kMUnknown : kUnknown);
        rItr->setBinContent(-1.);
        continue;
      }

      float mean(tItr->getBinContent());
      float rms(tItr->getBinError() * sqrt(entries));

      MEs_[kMeanSM]->fill(id, mean);
      MEs_[kMeanAll]->fill(id, mean);
      MEs_[kProjEta]->fill(id, mean);
      MEs_[kProjPhi]->fill(id, mean);
      MEs_[kRMSAll]->fill(id, rms);
      rItr->setBinContent(rms);

      bool negative(false);
      float posTime(0.);

      if(id.subdetId() == EcalBarrel){
        EBDetId ebid(id);
        if(ebid.zside() < 0){
          negative = true;
          EBDetId posId(EBDetId::switchZSide(ebid));
          posTime = sources_[kTimeMap]->getBinContent(posId);
        }
      }
      else{
        EEDetId eeid(id);
        if(eeid.zside() < 0){
          negative = true;
          EEDetId posId(EEDetId::switchZSide(eeid));
          posTime = sources_[kTimeMap]->getBinContent(posId);
        }
      }
      if(negative){
        MEs_[kFwdBkwdDiff]->fill(id, posTime - mean);
        MEs_[kFwdvBkwd]->fill(id, mean, posTime);
      }

      if(abs(mean) > meanThresh || rms > rmsThresh)
        qItr->setBinContent(doMask ? kMBad : kBad);
      else
        qItr->setBinContent(doMask ? kMGood : kGood);
    }

    MESet::iterator qsEnd(MEs_[kQualitySummary]->end());

    for(MESet::iterator qsItr(MEs_[kQualitySummary]->beginChannel()); qsItr != qsEnd; qsItr.toNextChannel()){

      DetId tId(qsItr->getId());

      std::vector<DetId> ids;

      if(tId.subdetId() == EcalTriggerTower)
        ids = getTrigTowerMap()->constituentsOf(EcalTrigTowerDetId(tId));
      else
        ids = scConstituents(EcalScDetId(tId));

      int minTowerEntries(minTowerEntries_);
      float meanThresh(toleranceMean_);
      float rmsThresh(toleranceRMS_);

      if(isForward(tId)){
        minTowerEntries = minTowerEntriesFwd_;
          meanThresh = toleranceMeanFwd_;
          rmsThresh = toleranceRMSFwd_;
      }

      // tower entries != sum(channel entries) because of the difference in timing cut at the source
      float summaryEntries(sources_[kTimeAllMap]->getBinEntries(tId));

      float towerEntries(0.);
      float towerMean(0.);
      float towerMean2(0.);

      bool doMask(false);

      for(vector<DetId>::iterator idItr(ids.begin()); idItr != ids.end(); ++idItr){
        DetId& id(*idItr);

        doMask |= applyMask_(kQuality, id, mask);

        MESet::const_iterator tmItr(sources_[kTimeMap], id);

        float entries(tmItr->getBinEntries());
        if(entries < 0.) continue;
        towerEntries += entries;
        float mean(tmItr->getBinContent());
        towerMean += mean * entries;
        float rms(tmItr->getBinError() * sqrt(entries));
        towerMean2 += (rms * rms + mean * mean) * entries;
      }

      double quality(doMask ? kMUnknown : kUnknown);
      if(towerEntries / ids.size() > minTowerEntries / 25.){
        if(summaryEntries < towerEntries * (1. - tailPopulThreshold_)) // large timing deviation
          quality = doMask ? kMBad : kBad;
        else{
	  towerMean /= towerEntries;
	  towerMean2 /= towerEntries;

	  float towerRMS(sqrt(towerMean2 - towerMean * towerMean));

	  if(abs(towerMean) > meanThresh || towerRMS > rmsThresh)
	    quality = doMask ? kMBad : kBad;
          else
            quality = doMask ? kMGood : kGood;
        }
      }

      qsItr->setBinContent(quality);
    }
  }

  /*static*/
  void
  TimingClient::setMEOrdering(std::map<std::string, unsigned>& _nameToIndex)
  {
    _nameToIndex["Quality"] = kQuality;
    _nameToIndex["MeanSM"] = kMeanSM;
    _nameToIndex["MeanAll"] = kMeanAll;
    _nameToIndex["FwdBkwdDiff"] = kFwdBkwdDiff;
    _nameToIndex["FwdvBkwd"] = kFwdvBkwd;
    _nameToIndex["RMSMap"] = kRMSMap;
    _nameToIndex["RMSAll"] = kRMSAll;
    _nameToIndex["ProjEta"] = kProjEta;
    _nameToIndex["ProjPhi"] = kProjPhi;
    _nameToIndex["QualitySummary"] = kQualitySummary;

    _nameToIndex["TimeAllMap"] = kTimeAllMap;
    _nameToIndex["TimeMap"] = kTimeMap;
  }

  DEFINE_ECALDQM_WORKER(TimingClient);
}
