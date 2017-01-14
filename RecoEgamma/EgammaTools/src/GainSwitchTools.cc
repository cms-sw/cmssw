#include "RecoEgamma/EgammaTools/interface/GainSwitchTools.h"


#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"
#include "DataFormats/DetId/interface/DetId.h"
const std::vector<int> GainSwitchTools::gainSwitchFlags_={EcalRecHit::kHasSwitchToGain6,EcalRecHit::kHasSwitchToGain1};

 //this should really live in EcalClusterTools
int GainSwitchTools::nrCrysWithFlagsIn5x5(const DetId& id,const std::vector<int>& flags,const EcalRecHitCollection* recHits,const CaloTopology *topology)
{
  int nrFound=0;
  CaloNavigator<DetId> cursor = CaloNavigator<DetId>( id, topology->getSubdetectorTopology( id ) );
  
  for ( int eastNr = -2; eastNr <= 2; ++eastNr ) { //east is eta in barrel
    for ( int northNr = -2; northNr <= 2; ++northNr ) { //north is phi in barrel
      cursor.home();
      cursor.offsetBy( eastNr, northNr);
      DetId id = *cursor;
      auto recHitIt = recHits->find(id);
      if(recHitIt!=recHits->end() && 
	 recHitIt->checkFlags(flags)){
	nrFound++;
      }
    }
  }
  return nrFound;
}

bool GainSwitchTools::hasEBGainSwitch(const reco::SuperCluster& superClus,const EcalRecHitCollection* recHits)
{
  if(!recHits || superClus.seed()->seed().subdetId()!=EcalBarrel) return false;
  for(const auto & clus : superClus.clusters()){
    for(const auto& hit : clus->hitsAndFractions()){
      auto recHitIt = recHits->find(hit.first);
      if(recHitIt!=recHits->end() && 
	 recHitIt->checkFlags(gainSwitchFlags())){
	return true;
      }
    }
  }
  return false;	
}

bool GainSwitchTools::hasEBGainSwitchIn5x5(const reco::SuperCluster& superClus,const EcalRecHitCollection* recHits,const CaloTopology *topology)
{
  if(recHits || superClus.seed()->seed().subdetId()!=EcalBarrel) return nrCrysWithFlagsIn5x5(superClus.seed()->seed(),gainSwitchFlags(),recHits,topology)!=0;
  else return false;
}

bool GainSwitchTools::hasEBGainSwitch(const EcalRecHitCollection* recHits)
{
  if(!recHits) return false;
  for(auto hit : *recHits){
    if(hit.id().subdetId()==EcalBarrel && hit.checkFlags(gainSwitchFlags())) return true;
  }
  return false;
}
  


