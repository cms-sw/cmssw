#include "../interface/OccupancyClient.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

namespace ecaldqm {

  OccupancyClient::OccupancyClient(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerClient(_workerParams, _commonParams, "OccupancyClient"),
    minHits_(_workerParams.getUntrackedParameter<int>("minHits")),
    deviationThreshold_(_workerParams.getUntrackedParameter<double>("deviationThreshold"))
  {
    qualitySummaries_.insert(kQualitySummary);
  }

  void
  OccupancyClient::producePlots()
  {
    using namespace std;

    MEs_[kHotDigi]->reset();
    MEs_[kHotRecHitThr]->reset();
    MEs_[kHotTPDigiThr]->reset();

    uint32_t mask(1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_RMS_ERROR |
                  1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_WARNING |
                  1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_ERROR);

    vector<double> digiPhiRingMean(56, 0.);
    vector<double> rechitPhiRingMean(56, 0.);
    vector<int> numCrystals(56, 0); // this is static, but is easier to count now

    MESet::const_iterator dEnd(sources_[kDigi]->end());
    MESet::const_iterator rItr(sources_[kRecHitThr]);
    for(MESet::const_iterator dItr(sources_[kDigi]->beginChannel()); dItr != dEnd; dItr.toNextChannel()){
      rItr = dItr;

      float entries(dItr->getBinContent());
      float rhentries(rItr->getBinContent());

      DetId id(dItr->getId());
      int ieta(0);
      if(id.subdetId() == EcalTriggerTower) // barrel
        ieta = EcalTrigTowerDetId(id).ieta();
      else{
        std::vector<DetId> ids(scConstituents(EcalScDetId(id)));
        if(ids.size() == 0) continue;
        ieta = getTrigTowerMap()->towerOf(ids[0]).ieta();
      }

      unsigned index(ieta < 0 ? ieta + 28 : ieta + 27);

      digiPhiRingMean.at(index) += entries;
      rechitPhiRingMean.at(index) += rhentries;
      numCrystals.at(index) += 1;
    }

    for(int ie(0); ie < 56; ie++){
      digiPhiRingMean[ie] /= numCrystals[ie];
      rechitPhiRingMean[ie] /= numCrystals[ie];
    }

    // second round to find hot towers
    for(MESet::const_iterator dItr(sources_[kDigi]->beginChannel()); dItr != dEnd; dItr.toNextChannel()){
      DetId id(dItr->getId());

      bool doMask(applyMask_(kQualitySummary, id, mask));

      rItr = dItr;

      float entries(dItr->getBinContent());
      float rhentries(rItr->getBinContent());

      int ieta(0);
      if(id.subdetId() == EcalTriggerTower) // barrel
        ieta = EcalTrigTowerDetId(id).ieta();
      else{
        std::vector<DetId> ids(scConstituents(EcalScDetId(id)));
        if(ids.size() == 0) continue;
        ieta = getTrigTowerMap()->towerOf(ids[0]).ieta();
      }

      unsigned index(ieta < 0 ? ieta + 28 : ieta + 27);

      int quality(doMask ? kMGood : kGood);

      if(entries > minHits_ && entries > digiPhiRingMean.at(index) * deviationThreshold_){
        MEs_[kHotDigi]->fill(id);
        quality = doMask ? kMBad : kBad;
      }
      if(rhentries > minHits_ && rhentries > rechitPhiRingMean.at(index) * deviationThreshold_){
        MEs_[kHotRecHitThr]->fill(id);
        quality = doMask ? kMBad : kBad;
      }

      MEs_[kQualitySummary]->setBinContent(id, double(quality));
    }

    vector<double> tpdigiPhiRingMean(56, 0.);

    for(unsigned iTT(0); iTT < EcalTrigTowerDetId::kSizeForDenseIndexing; ++iTT){
      EcalTrigTowerDetId ttid(EcalTrigTowerDetId::detIdFromDenseIndex(iTT));
      float entries(sources_[kTPDigiThr]->getBinContent(ttid));

      unsigned index(ttid.ieta() < 0 ? ttid.ieta() + 28 : ttid.ieta() + 27);
      tpdigiPhiRingMean.at(index) += entries;
    }

    for(int ie(0); ie < 28; ie++){
      float denom(-1.);
      if(ie < 27) denom = 72.;
      else denom = 36.;
      tpdigiPhiRingMean[ie] /= denom;
      tpdigiPhiRingMean[55 - ie] /= denom;
    }

    for(unsigned iTT(0); iTT < EcalTrigTowerDetId::kSizeForDenseIndexing; ++iTT){
      EcalTrigTowerDetId ttid(EcalTrigTowerDetId::detIdFromDenseIndex(iTT));

      float entries(sources_[kTPDigiThr]->getBinContent(ttid));

      int quality(kGood);

      if(entries > minHits_ && entries > tpdigiPhiRingMean.at(ttid.ietaAbs() - 1) * deviationThreshold_){
        MEs_[kHotTPDigiThr]->fill(ttid);
        quality = kBad;
      }

      if(quality != kBad) continue;

      std::vector<DetId> ids(getTrigTowerMap()->constituentsOf(ttid));
      for(unsigned iD(0); iD < ids.size(); ++iD){
        DetId& id(ids[iD]);

        int quality(MEs_[kQualitySummary]->getBinContent(id));
        if(quality == kMBad || quality == kBad) continue;

        MEs_[kQualitySummary]->setBinContent(id, applyMask_(kQualitySummary, id, mask) ? kMBad : kBad);
      }
    }
  }

  /*static*/
  void
  OccupancyClient::setMEOrdering(std::map<std::string, unsigned>& _nameToIndex)
  {
    _nameToIndex["HotDigi"] = kHotDigi;
    _nameToIndex["HotRecHitThr"] = kHotRecHitThr;
    _nameToIndex["HotTPDigiThr"] = kHotTPDigiThr;
    _nameToIndex["QualitySummary"] = kQualitySummary;

    _nameToIndex["Digi"] = kDigi;
    _nameToIndex["RecHitThr"] = kRecHitThr;
    _nameToIndex["TPDigiThr"] = kTPDigiThr;
  }

  DEFINE_ECALDQM_WORKER(OccupancyClient);
}
