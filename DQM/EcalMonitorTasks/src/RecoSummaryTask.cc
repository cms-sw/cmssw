#include "../interface/RecoSummaryTask.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalTools.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace ecaldqm
{
  RecoSummaryTask::RecoSummaryTask() :
    DQWorkerTask(),
    rechitThresholdEB_(0.),
    rechitThresholdEE_(0.),
    ebHits_(0),
    eeHits_(0)
  {
  }

  void
  RecoSummaryTask::setParams(edm::ParameterSet const& _params)
  {
    rechitThresholdEB_ = _params.getUntrackedParameter<double>("rechitThresholdEB");
    rechitThresholdEE_ = _params.getUntrackedParameter<double>("rechitThresholdEE");
  }

  void
  RecoSummaryTask::addDependencies(DependencySet& _dependencies)
  {
    _dependencies.push_back(Dependency(kEBBasicCluster, kEBRecHit));
    _dependencies.push_back(Dependency(kEEBasicCluster, kEERecHit));
  }
  
  bool
  RecoSummaryTask::filterRunType(short const* _runType)
  {
    for(unsigned iFED(0); iFED != ecaldqm::nDCC; iFED++){
      if(_runType[iFED] == EcalDCCHeaderBlock::COSMIC ||
         _runType[iFED] == EcalDCCHeaderBlock::MTCC ||
         _runType[iFED] == EcalDCCHeaderBlock::COSMICS_GLOBAL ||
         _runType[iFED] == EcalDCCHeaderBlock::PHYSICS_GLOBAL ||
         _runType[iFED] == EcalDCCHeaderBlock::COSMICS_LOCAL ||
         _runType[iFED] == EcalDCCHeaderBlock::PHYSICS_LOCAL) return true;
    }

    return false;
  }

  void
  RecoSummaryTask::endEvent(edm::Event const&, edm::EventSetup const&)
  {
    ebHits_ = 0;
    eeHits_ = 0;
  }

  void 
  RecoSummaryTask::runOnRecHits(EcalRecHitCollection const& _hits, Collections _collection)
  {
    bool isBarrel(_collection == kEBRecHit);

    MESet& meEnergyMax(MEs_.at("EnergyMax"));
    MESet& meChi2(MEs_.at("Chi2"));
    MESet& meTime(MEs_.at("Time"));
    MESet* meSwissCross(isBarrel ? &MEs_.at("SwissCross") : 0);
    MESet& meRecoFlag(MEs_.at("RecoFlagAll"));

    double maxE[2] = {-1. -1};
    int subdet(isBarrel ? EcalBarrel : EcalEndcap);
    
    for(EcalRecHitCollection::const_iterator hitItr(_hits.begin()); hitItr != _hits.end(); ++hitItr){
      meRecoFlag.fill(subdet, hitItr->recoFlag());
      float energy(hitItr->energy());

      int signedSubdet;
      float rechitThreshold;

      if(isBarrel){
        signedSubdet = EcalBarrel;
        rechitThreshold = rechitThresholdEB_;

        if(energy > 3.){
          EBDetId ebId(hitItr->id());
          if(ebId.ieta() != 85)
            meSwissCross->fill(EcalTools::swissCross(ebId, _hits, 0.));
        }

        if(energy > maxE[0]) maxE[0] = energy;
      }
      else{
        rechitThreshold = rechitThresholdEE_;

        EEDetId eeId(hitItr->id());
        if(eeId.zside() < 0){
          signedSubdet = -EcalEndcap;
          if(energy > maxE[0]) maxE[0] = energy;
        }
        else{
          signedSubdet = EcalEndcap;
          if(energy > maxE[1]) maxE[1] = energy;
        }
      }

      if(energy > rechitThreshold){
        meChi2.fill(signedSubdet, hitItr->chi2());
        meTime.fill(signedSubdet, hitItr->time());
      }
    }

    if(isBarrel){
      meEnergyMax.fill(EcalBarrel, maxE[0]);

      ebHits_ = &_hits;
    }
    else{
      meEnergyMax.fill(-EcalEndcap, maxE[0]);
      meEnergyMax.fill(EcalEndcap, maxE[1]);

      eeHits_ = &_hits;
    }
  }

  void 
  RecoSummaryTask::runOnReducedRecHits(EcalRecHitCollection const& _hits, Collections _collections)
  {
    MESet& meRecoFlag(MEs_.at("RecoFlagReduced"));

    int subdet(_collections == kEBReducedRecHit ? EcalBarrel : EcalEndcap);

    for(EcalRecHitCollection::const_iterator hitItr(_hits.begin()); hitItr != _hits.end(); ++hitItr)
      meRecoFlag.fill(subdet, hitItr->recoFlag());
  }

  void
  RecoSummaryTask::runOnBasicClusters(edm::View<reco::CaloCluster> const& _bcs, Collections _collection)
  {
    bool isBarrel(_collection == kEBBasicCluster);

    MESet& meRecoFlag(MEs_.at("RecoFlagBasicCluster"));

    EcalRecHitCollection const* hitCol(isBarrel ? ebHits_ : eeHits_);
    int subdet(isBarrel ? EcalBarrel : EcalEndcap);

    for(edm::View<reco::CaloCluster>::const_iterator bcItr(_bcs.begin()); bcItr != _bcs.end(); ++bcItr){
      if(bcItr->caloID().detectors() != 0){
        if(isBarrel && !bcItr->caloID().detector(reco::CaloID::DET_ECAL_BARREL)) continue;
        if(!isBarrel && !bcItr->caloID().detector(reco::CaloID::DET_ECAL_ENDCAP)) continue;
      }

      std::vector<std::pair<DetId, float> > const& haf(bcItr->hitsAndFractions());
      for(unsigned iH(0); iH != haf.size(); ++iH){
        if(isBarrel && haf[iH].first.subdetId() != EcalBarrel) continue;
        if(!isBarrel && haf[iH].first.subdetId() != EcalEndcap) continue;
        EcalRecHitCollection::const_iterator hItr(hitCol->find(haf[iH].first));
        if(hItr == hitCol->end()) continue;
        meRecoFlag.fill(subdet, hItr->recoFlag());
      }
    }
  }

  DEFINE_ECALDQM_WORKER(RecoSummaryTask);
}
