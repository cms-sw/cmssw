#include "../interface/OccupancyClient.h"
#include "../interface/EcalDQMClientUtils.h"

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
    qualitySummaries_.insert("QualitySummary");
  }

  void
  OccupancyClient::producePlots()
  {
    using namespace std;

    MESet* meQualitySummary(MEs_["QualitySummary"]);
    MESet* meHotDigi(MEs_["HotDigi"]);
    MESet* meHotRecHitThr(MEs_["HotRecHitThr"]);
    MESet* meHotTPDigiThr(MEs_["HotTPDigiThr"]);

    MESet const* sDigi(sources_["Digi"]);
    MESet const* sRecHitThr(sources_["RecHitThr"]);
    MESet const* sTPDigiThr(sources_["TPDigiThr"]);

    meHotDigi->reset();
    meHotRecHitThr->reset();
    meHotTPDigiThr->reset();

    uint32_t mask(1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_RMS_ERROR |
                  1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_WARNING |
                  1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_ERROR);

    vector<double> digiPhiRingMean(56, 0.);
    vector<double> rechitPhiRingMean(56, 0.);
    vector<int> numCrystals(56, 0); // this is static, but is easier to count now

    MESet::const_iterator dEnd(sDigi->end());
    MESet::const_iterator rItr(sRecHitThr);
    for(MESet::const_iterator dItr(sDigi->beginChannel()); dItr != dEnd; dItr.toNextChannel()){
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
    for(MESet::const_iterator dItr(sDigi->beginChannel()); dItr != dEnd; dItr.toNextChannel()){
      DetId id(dItr->getId());

      bool doMask(applyMask(meQualitySummary->getBinType(), id, mask));

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
        meHotDigi->fill(id);
        quality = doMask ? kMBad : kBad;
      }
      if(rhentries > minHits_ && rhentries > rechitPhiRingMean.at(index) * deviationThreshold_){
        meHotRecHitThr->fill(id);
        quality = doMask ? kMBad : kBad;
      }

      meQualitySummary->setBinContent(id, double(quality));
    }

    vector<double> tpdigiPhiRingMean(56, 0.);

    for(unsigned iTT(0); iTT < EcalTrigTowerDetId::kSizeForDenseIndexing; ++iTT){
      EcalTrigTowerDetId ttid(EcalTrigTowerDetId::detIdFromDenseIndex(iTT));
      float entries(sTPDigiThr->getBinContent(ttid));

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

      float entries(sTPDigiThr->getBinContent(ttid));

      int quality(kGood);

      if(entries > minHits_ && entries > tpdigiPhiRingMean.at(ttid.ietaAbs() - 1) * deviationThreshold_){
        meHotTPDigiThr->fill(ttid);
        quality = kBad;
      }

      if(quality != kBad) continue;

      std::vector<DetId> ids(getTrigTowerMap()->constituentsOf(ttid));
      for(unsigned iD(0); iD < ids.size(); ++iD){
        DetId& id(ids[iD]);

        int quality(meQualitySummary->getBinContent(id));
        if(quality == kMBad || quality == kBad) continue;

        meQualitySummary->setBinContent(id, applyMask(meQualitySummary->getBinType(), id, mask) ? kMBad : kBad);
      }
    }
  }

  DEFINE_ECALDQM_WORKER(OccupancyClient);
}
