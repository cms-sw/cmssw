
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronHcalHelper.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHcalIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTowerIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHadTower.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace reco ;

ElectronHcalHelper::ElectronHcalHelper( const Configuration & cfg  )
  : cfg_(cfg), caloGeomCacheId_(0), hcalIso_(0), towerIso1_(0), towerIso2_(0),hadTower_(0) 
{  }

void ElectronHcalHelper::checkSetup( const edm::EventSetup & es )
 {

  if (cfg_.hOverEConeSize==0)
   { return ; }

  if (cfg_.useTowers)
   {
    delete hadTower_ ;
    hadTower_ = new EgammaHadTower(es) ;
   }
  else
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

  if (cfg_.useTowers)
   {
    delete towerIso1_ ; towerIso1_ = 0 ;
    delete towerIso2_ ; towerIso2_ = 0 ;

    edm::Handle<CaloTowerCollection> towersH_ ;
    if (!evt.getByToken(cfg_.hcalTowers,towersH_)){
      edm::LogError("ElectronHcalHelper::readEvent")
	<<"failed to get the hcal towers"; 
    }
    hadTower_->setTowerCollection(towersH_.product());
    towerIso1_ = new EgammaTowerIsolation(cfg_.hOverEConeSize,0.,cfg_.hOverEPtMin,1,towersH_.product()) ;
    towerIso2_ = new EgammaTowerIsolation(cfg_.hOverEConeSize,0.,cfg_.hOverEPtMin,2,towersH_.product()) ;
   }
  else
   {
    delete hcalIso_ ; hcalIso_ = 0 ;

    edm::Handle<HBHERecHitCollection> hbhe_;
    if (!evt.getByToken(cfg_.hcalRecHits,hbhe_)) { 
      edm::LogError("ElectronHcalHelper::readEvent")
	<<"failed to get the rechits";
    }

    hcalIso_ = new EgammaHcalIsolation(cfg_.hOverEConeSize,0.,cfg_.hOverEHBMinE,cfg_.hOverEHFMinE,0.,0.,caloGeom_, *hbhe_) ;
   }
 }

std::vector<CaloTowerDetId> ElectronHcalHelper::hcalTowersBehindClusters( const reco::SuperCluster & sc )
 { return hadTower_->towersOf(sc) ; }

double ElectronHcalHelper::hcalESumDepth1BehindClusters( const std::vector<CaloTowerDetId> & towers )
 { return hadTower_->getDepth1HcalESum(towers) ; }

double ElectronHcalHelper::hcalESumDepth2BehindClusters( const std::vector<CaloTowerDetId> & towers )
 { return hadTower_->getDepth2HcalESum(towers) ; }

double ElectronHcalHelper::hcalESum( const SuperCluster & sc, const std::vector<CaloTowerDetId > * excludeTowers )
 {
  if (cfg_.hOverEConeSize==0)
   { return 0 ; }
  if (cfg_.useTowers)
    { return(hcalESumDepth1(sc,excludeTowers)+hcalESumDepth2(sc,excludeTowers)) ; }
  else
   { return hcalIso_->getHcalESum(&sc) ; }
 }

double ElectronHcalHelper::hcalESumDepth1( const SuperCluster & sc ,const std::vector<CaloTowerDetId > * excludeTowers )
 {
  if (cfg_.hOverEConeSize==0)
   { return 0 ; }
  if (cfg_.useTowers)
    { return towerIso1_->getTowerESum(&sc, excludeTowers) ; }
  else
   { return hcalIso_->getHcalESumDepth1(&sc) ; }
 }

double ElectronHcalHelper::hcalESumDepth2( const SuperCluster & sc ,const std::vector<CaloTowerDetId > * excludeTowers  )
 {
  if (cfg_.hOverEConeSize==0)
   { return 0 ; }
  if (cfg_.useTowers)
    { return towerIso2_->getTowerESum(&sc, excludeTowers) ; }
  else
   { return hcalIso_->getHcalESumDepth2(&sc) ; }
 }

ElectronHcalHelper::~ElectronHcalHelper()
 {
  if (cfg_.hOverEConeSize==0)
   { return ; }
  if (cfg_.useTowers)
   {
    delete towerIso1_ ;
    delete towerIso2_ ;
    delete hadTower_;
   }
  else
   {
    delete hcalIso_ ;
   }
 }


