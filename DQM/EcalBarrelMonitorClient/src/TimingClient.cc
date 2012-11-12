#include "../interface/TimingClient.h"
#include "../interface/EcalDQMClientUtils.h"

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
    qualitySummaries_.insert("Quality");
    qualitySummaries_.insert("QualitySummary");
  }

  void
  TimingClient::producePlots()
  {
    using namespace std;

    MESet* meQuality(MEs_["Quality"]);
    MESet* meMeanSM(MEs_["MeanSM"]);
    MESet* meMeanAll(MEs_["MeanAll"]);
    MESet* meFwdBkwdDiff(MEs_["FwdBkwdDiff"]);
    MESet* meFwdvBkwd(MEs_["FwdvBkwd"]);
    MESet* meRMSMap(MEs_["RMSMap"]);
    MESet* meRMSAll(MEs_["RMSAll"]);
    MESet* meProjEta(MEs_["ProjEta"]);
    MESet* meProjPhi(MEs_["ProjPhi"]);
    MESet* meQualitySummary(MEs_["QualitySummary"]);

    MESet const* sTimeAllMap(sources_["TimeAllMap"]);
    MESet const* sTimeMap(sources_["TimeMap"]);

    meMeanSM->reset();
    meMeanAll->reset();
    meRMSAll->reset();
    meProjEta->reset();
    meProjPhi->reset();
    meFwdBkwdDiff->reset();
    meFwdvBkwd->reset();

    uint32_t mask(1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_WARNING);

    MESet::iterator qEnd(meQuality->end());

    MESet::iterator rItr(meRMSMap);
    MESet::const_iterator tItr(sTimeMap);

    for(MESet::iterator qItr(meQuality->beginChannel()); qItr != qEnd; qItr.toNextChannel()){

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

      bool doMask(applyMask(meQuality->getBinType(), id, mask));

      float entries(tItr->getBinEntries());

      if(entries < minChannelEntries){
        qItr->setBinContent(doMask ? kMUnknown : kUnknown);
        rItr->setBinContent(-1.);
        continue;
      }

      float mean(tItr->getBinContent());
      float rms(tItr->getBinError() * sqrt(entries));

      meMeanSM->fill(id, mean);
      meMeanAll->fill(id, mean);
      meProjEta->fill(id, mean);
      meProjPhi->fill(id, mean);
      meRMSAll->fill(id, rms);
      rItr->setBinContent(rms);

      bool negative(false);
      float posTime(0.);

      if(id.subdetId() == EcalBarrel){
        EBDetId ebid(id);
        if(ebid.zside() < 0){
          negative = true;
          EBDetId posId(EBDetId::switchZSide(ebid));
          posTime = sTimeMap->getBinContent(posId);
        }
      }
      else{
        EEDetId eeid(id);
        if(eeid.zside() < 0){
          negative = true;
          EEDetId posId(EEDetId::switchZSide(eeid));
          posTime = sTimeMap->getBinContent(posId);
        }
      }
      if(negative){
        meFwdBkwdDiff->fill(id, posTime - mean);
        meFwdvBkwd->fill(id, mean, posTime);
      }

      if(abs(mean) > meanThresh || rms > rmsThresh)
        qItr->setBinContent(doMask ? kMBad : kBad);
      else
        qItr->setBinContent(doMask ? kMGood : kGood);
    }

    MESet::iterator qsEnd(meQualitySummary->end());

    for(MESet::iterator qsItr(meQualitySummary->beginChannel()); qsItr != qsEnd; qsItr.toNextChannel()){

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
      float summaryEntries(sTimeAllMap->getBinEntries(tId));

      float towerEntries(0.);
      float towerMean(0.);
      float towerMean2(0.);

      bool doMask(false);

      for(vector<DetId>::iterator idItr(ids.begin()); idItr != ids.end(); ++idItr){
        DetId& id(*idItr);

        doMask |= applyMask(meQuality->getBinType(), id, mask);

        MESet::const_iterator tmItr(sTimeMap, id);

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

  DEFINE_ECALDQM_WORKER(TimingClient);
}

