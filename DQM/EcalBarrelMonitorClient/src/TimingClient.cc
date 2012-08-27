#include "../interface/TimingClient.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

#include <cmath>

namespace ecaldqm {

  TimingClient::TimingClient(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerClient(_workerParams, _commonParams, "TimingClient"),
    expectedMean_(_workerParams.getUntrackedParameter<double>("expectedMean")),
    meanThreshold_(_workerParams.getUntrackedParameter<double>("meanThreshold")),
    rmsThreshold_(_workerParams.getUntrackedParameter<double>("rmsThreshold")),
    minChannelEntries_(_workerParams.getUntrackedParameter<int>("minChannelEntries")),
    minTowerEntries_(_workerParams.getUntrackedParameter<int>("minTowerEntries")),
    tailPopulThreshold_(_workerParams.getUntrackedParameter<double>("tailPopulThreshold"))
  {
  }

  void
  TimingClient::bookMEs()
  {
    DQWorker::bookMEs();

    MEs_[kQuality]->resetAll(-1.);
    MEs_[kRMSMap]->resetAll(-1.);
    MEs_[kQualitySummary]->resetAll(-1.);
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

      float entries(tItr->getBinEntries());

      if(entries < minChannelEntries_){
        qItr->setBinContent(maskQuality_(qItr, mask, 2));
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

      if(abs(mean - expectedMean_) > meanThreshold_ || rms > rmsThreshold_)
        qItr->setBinContent(maskQuality_(qItr, mask, 0));
      else
        qItr->setBinContent(maskQuality_(qItr, mask, 1));
    }

    MESet::iterator qsEnd(MEs_[kQualitySummary]->end());

    MESet::const_iterator taItr(sources_[kTimeAllMap]);

    for(MESet::iterator qsItr(MEs_[kQualitySummary]->beginChannel()); qsItr != qsEnd; qsItr.toNextChannel()){

      DetId tId(qsItr->getId());

      taItr = qsItr;

      // tower entries != sum(channel entries) because of the difference in timing cut at the source
      float summaryEntries(taItr->getBinEntries());

      std::vector<DetId> ids;

      if(tId.subdetId() == EcalTriggerTower)
        ids = getTrigTowerMap()->constituentsOf(EcalTrigTowerDetId(tId));
      else{
        std::pair<int, int> dccsc(getElectronicsMap()->getDCCandSC(EcalScDetId(tId)));
        ids = getElectronicsMap()->dccTowerConstituents(dccsc.first, dccsc.second);
      }

      float towerEntries(0.);
      float towerMean(0.);
      float towerMean2(0.);

      for(vector<DetId>::iterator idItr(ids.begin()); idItr != ids.end(); ++idItr){
        DetId& id(*idItr);

        MESet::const_iterator tmItr(sources_[kTimeMap], id);

        towerEntries += tmItr->getBinEntries();
        float mean(tmItr->getBinContent());
        towerMean += mean;
        towerMean2 += mean * mean;
      }

      int quality(2);
      if(towerEntries > minTowerEntries_){
        if(summaryEntries < towerEntries * (1. - tailPopulThreshold_)) // large timing deviation
          quality = 0;
        else{
	  towerMean /= ids.size();
	  towerMean2 /= ids.size();

	  float towerRMS(0.);
	  float variance(towerMean2 - towerMean * towerMean);
	  if(variance > 0.) towerRMS = sqrt(variance);

	  if(abs(towerMean - expectedMean_) > meanThreshold_ || towerRMS > rmsThreshold_)
	    quality = 0;
          else
            quality = 1;
        }
      }

      qsItr->setBinContent(maskQuality_(qsItr, mask, quality));

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
