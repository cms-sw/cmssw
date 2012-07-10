
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronHcalHelper.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHcalIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTowerIsolation.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace reco ;

ElectronHcalHelper::ElectronHcalHelper( const Configuration & cfg  )
 : cfg_(cfg), caloGeomCacheId_(0), hbhe_(0), mhbhe_(0), hcalIso_(0), towersH_(0), towerIso1_(0), towerIso2_(0)
 {}

void ElectronHcalHelper::checkSetup( const edm::EventSetup & es )
 {
  if (cfg_.hOverEConeSize==0)
   { return ; }

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

void ElectronHcalHelper::readEvent( edm::Event & evt )
 {
  if (cfg_.hOverEConeSize==0)
   { return ; }

  if (cfg_.useTowers)
   {
    delete towerIso1_ ; towerIso1_ = 0 ;
    delete towerIso2_ ; towerIso2_ = 0 ;
    delete towersH_ ; towersH_ = 0 ;

    towersH_ = new edm::Handle<CaloTowerCollection>() ;
    if (!evt.getByLabel(cfg_.hcalTowers,*towersH_))
     { edm::LogError("ElectronHcalHelper::readEvent")<<"failed to get the hcal towers of label "<<cfg_.hcalTowers ; }
    towerIso1_ = new EgammaTowerIsolation(cfg_.hOverEConeSize,0.,cfg_.hOverEPtMin,1,towersH_->product()) ;
    towerIso2_ = new EgammaTowerIsolation(cfg_.hOverEConeSize,0.,cfg_.hOverEPtMin,2,towersH_->product()) ;
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

double ElectronHcalHelper::hcalESum( const SuperCluster & sc )
 {
  if (cfg_.hOverEConeSize==0)
   { return 0 ; }
  if (cfg_.useTowers)
   { return(hcalESumDepth1(sc)+hcalESumDepth2(sc)) ; }
  else
   { return hcalIso_->getHcalESum(&sc) ; }
 }

double ElectronHcalHelper::hcalESumDepth1( const SuperCluster & sc )
 {
  if (cfg_.hOverEConeSize==0)
   { return 0 ; }
  if (cfg_.useTowers)
   { return towerIso1_->getTowerESum(&sc) ; }
  else
   { return hcalIso_->getHcalESumDepth1(&sc) ; }
 }

double ElectronHcalHelper::hcalESumDepth2( const SuperCluster & sc )
 {
  if (cfg_.hOverEConeSize==0)
   { return 0 ; }
  if (cfg_.useTowers)
   { return towerIso2_->getTowerESum(&sc) ; }
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
    delete towersH_ ;
   }
  else
   {
    delete hcalIso_ ;
    delete mhbhe_ ;
    delete hbhe_ ;
   }
 }


