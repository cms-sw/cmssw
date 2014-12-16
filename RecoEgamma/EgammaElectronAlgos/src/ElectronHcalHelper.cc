
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronHcalHelper.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHcalIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTowerIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHadTower.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace reco ;

ElectronHcalHelper::ElectronHcalHelper( const Configuration & cfg  )
  : cfg_(cfg), caloGeomCacheId_(0), hbhe_(0), mhbhe_(0), hcalIso_(0), towersH_(0), towerIso1_(0), towerIso2_(0),hadTower_(0),hcalClusters_(0)
 {}

void ElectronHcalHelper::checkSetup( const edm::EventSetup & es )
 {

  if (cfg_.hOverEConeSize==0)
   { return ; }

  if(hadTower_) delete hadTower_ ;
  if(cfg_.hOverEMethod<=1) {
     hadTower_ = new EgammaHadTower(es,EgammaHadTower::SingleTower) ;
     //    std::cout << "ElectronHcalHelper, mode " << cfg_.hOverEMethod << std::endl;
    }
  if(cfg_.hOverEMethod==2) {
    hadTower_ = new EgammaHadTower(es,EgammaHadTower::TowersBehindCluster) ;
    //    std::cout << "ElectronHcalHelper, mode " << cfg_.hOverEMethod << std::endl;
  }
  if(cfg_.hOverEMethod==3) {
    hadTower_ = new EgammaHadTower(es,EgammaHadTower::HCALCluster) ;
  }

   
  if (!cfg_.useTowers)
   {
    unsigned long long newCaloGeomCacheId_
     = es.get<CaloGeometryRecord>().cacheIdentifier() ;
    if (caloGeomCacheId_!=newCaloGeomCacheId_)
     {
      caloGeomCacheId_ = newCaloGeomCacheId_ ;
      es.get<CaloGeometryRecord>().get(caloGeom_) ;
     }
   }
 }

void ElectronHcalHelper::readEvent( const edm::Event & evt )
 {
  if (cfg_.hOverEConeSize==0)
   { return ; }

  if (cfg_.hOverEMethod==3) 
   {
     delete hcalClusters_ ; hcalClusters_ = 0 ;
     hcalClusters_ = new edm::Handle<reco::PFClusterCollection>() ;
     if (!evt.getByLabel(cfg_.hcalClusters,*hcalClusters_))
     { edm::LogError("ElectronHcalHelper::readEvent")<<"failed to get the HCAL PF clusters "<<cfg_.hcalClusters ; }
     if(hcalClusters_) hadTower_->setHCALClusterCollection(hcalClusters_->product()); 
   }
  
  if (cfg_.useTowers)
   {
    delete towerIso1_ ; towerIso1_ = 0 ;
    delete towerIso2_ ; towerIso2_ = 0 ;
    delete towersH_ ; towersH_ = 0 ;

    towersH_ = new edm::Handle<CaloTowerCollection>() ;
    if (!evt.getByLabel(cfg_.hcalTowers,*towersH_))
     { edm::LogError("ElectronHcalHelper::readEvent")<<"failed to get the hcal towers of label "<<cfg_.hcalTowers ; }
    if(hadTower_) hadTower_->setTowerCollection(towersH_->product()); 
    //    towerIso1_ = new EgammaTowerIsolation(cfg_.hOverEConeSize,0.,cfg_.hOverEPtMin,1,towersH_->product()) ;
    //    towerIso2_ = new EgammaTowerIsolation(cfg_.hOverEConeSize,0.,cfg_.hOverEPtMin,2,towersH_->product()) ;
    // Set hOverETPtMin to 0 otherwise it will crash because of an assert
    towerIso1_ = new EgammaTowerIsolation(cfg_.hOverEConeSize,0.,0.,1,towersH_->product()) ;
    towerIso2_ = new EgammaTowerIsolation(cfg_.hOverEConeSize,0.,0.,2,towersH_->product()) ;
   }
  else
   {
    delete hcalIso_ ; hcalIso_ = 0 ;
    delete mhbhe_ ; mhbhe_ = 0 ;
    delete hbhe_ ; hbhe_ = 0 ;

    hbhe_=  new edm::Handle<HBHERecHitCollection>() ;
    if (!evt.getByLabel(cfg_.hcalRecHits,*hbhe_))
     { edm::LogError("ElectronHcalHelper::readEvent")<<"failed to get the rechits of label "<<cfg_.hcalRecHits ; }
    mhbhe_=  new HBHERecHitMetaCollection(**hbhe_) ;
    hcalIso_ = new EgammaHcalIsolation(cfg_.hOverEConeSize,0.,cfg_.hOverEHBMinE,cfg_.hOverEHFMinE,0.,0.,caloGeom_,mhbhe_) ;
   }

 }

std::vector<CaloTowerDetId> ElectronHcalHelper::hcalTowersBehindClusters( const reco::SuperCluster & sc )
{ 
  return hadTower_->towersOf(sc) ; }

double ElectronHcalHelper::hcalESumDepth1BehindClusters( const std::vector<CaloTowerDetId> & towers )
{ return hadTower_->getDepth1HcalESum(towers, cfg_.hOverEPtMin) ; }

double ElectronHcalHelper::hcalESumDepth2BehindClusters( const std::vector<CaloTowerDetId> & towers )
{ return hadTower_->getDepth2HcalESum(towers, cfg_.hOverEPtMin) ; }

double ElectronHcalHelper::hcalESumCone( const SuperCluster & sc, const std::vector<CaloTowerDetId > * excludeTowers )
 {
  if (cfg_.hOverEConeSize==0)
   { return 0 ; }
  if (cfg_.useTowers)
    { return(hcalESumDepth1(sc,excludeTowers)+hcalESumDepth2(sc,excludeTowers)) ; }
  else
   { return hcalIso_->getHcalESum(&sc) ; }
 }

double ElectronHcalHelper::hcalESumDepth1Cone( const SuperCluster & sc ,const std::vector<CaloTowerDetId > * excludeTowers )
 {
  if (cfg_.hOverEConeSize==0)
   { return 0 ; }
  if (cfg_.useTowers)
    { return towerIso1_->getTowerESum(&sc, excludeTowers) ; }
  else
   { return hcalIso_->getHcalESumDepth1(&sc) ; }
 }

double ElectronHcalHelper::hcalESumDepth2Cone( const SuperCluster & sc ,const std::vector<CaloTowerDetId > * excludeTowers  )
 {
  if (cfg_.hOverEConeSize==0)
   { return 0 ; }
  if (cfg_.useTowers)
    { return towerIso2_->getTowerESum(&sc, excludeTowers) ; }
  else
   { return hcalIso_->getHcalESumDepth2(&sc) ; }
 }

double ElectronHcalHelper::hcalESum( const reco::SuperCluster & sc, const std::vector<CaloTowerDetId> * excludeTowers ) {
  if(cfg_.hOverEMethod==0) {
    //    std::cout << "ElectronHcalHelper::hcalESum - default" << std::endl;
    return hcalESumCone(sc,excludeTowers);
  }
  // other cases
  std::vector<CaloTowerDetId> towers(hcalTowersBehindClusters(sc));
  // Et threshold is applied in these sums
  //  std::cout << "ElectronHcalHelper::hcalESum - FB" << std::endl;
  return hcalESumDepth1BehindClusters(towers) + hcalESumDepth2BehindClusters(towers);  
}

double ElectronHcalHelper::hcalESumDepth1( const reco::SuperCluster & sc, const std::vector<CaloTowerDetId> * excludeTowers ) {
  if(cfg_.hOverEMethod==0) {
    //    std::cout << "ElectronHcalHelper::hcalESumDepth1 - default " << std::endl;
    return hcalESumDepth1Cone(sc, excludeTowers);  
  }
  // Et threshold is applied in this sum
  //  std::cout << "ElectronHcalHelper::hcalESumDepth1 - FB " << std::endl;
  return hcalESumDepth1BehindClusters(hcalTowersBehindClusters(sc));
}

double ElectronHcalHelper::hcalESumDepth2( const reco::SuperCluster & sc, const std::vector<CaloTowerDetId> * excludeTowers ) {
  if(cfg_.hOverEMethod==0)     {
    //      std::cout << "ElectronHcalHelper::hcalESumDepth2 - default " << std::endl;
      return hcalESumDepth2Cone(sc, excludeTowers);  
    }
  // Et threshold is applied in this sum
  //  std::cout << "ElectronHcalHelper::hcalESumDepth2 - FB " << std::endl;
  return hcalESumDepth2BehindClusters(hcalTowersBehindClusters(sc));
}

double ElectronHcalHelper::HCALClustersBehindSC( const reco::SuperCluster & sc ) {
  return hadTower_->getHCALClusterEnergy(sc, 0.,cfg_.hOverEConeSize);
}

ElectronHcalHelper::~ElectronHcalHelper()
 {
  if (cfg_.hOverEConeSize==0)
   { return ; }
  if (cfg_.useTowers)
   {
    delete towerIso1_ ;
    delete towerIso2_ ;
    delete towersH_ ;
    delete hadTower_;
   }
  else
   {
    delete hcalIso_ ;
    delete mhbhe_ ;
    delete hbhe_ ;
   }
 }


