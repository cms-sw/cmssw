
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronHcalHelper.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHcalIsolation.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTowerIsolation.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

using namespace reco ;

ElectronHcalHelper::ElectronHcalHelper( const edm::ParameterSet & conf, bool useTowers, bool forPflow )
 : useTowers_(useTowers), caloGeomCacheId_(0), hbhe_(0), mhbhe_(0), hcalIso_(0), towersH_(0), towerIso1_(0), towerIso2_(0)
 {
  if (!forPflow)
   { hOverEConeSize_ = conf.getParameter<double>("hOverEConeSize") ; }
  else
   { hOverEConeSize_ = conf.getParameter<double>("hOverEConeSizePflow") ; }

  if (useTowers_)
   {
    hcalTowers_ = conf.getParameter<edm::InputTag>("hcalTowers") ;
    if (!forPflow)
     { hOverEPtMin_ = conf.getParameter<double>("hOverEPtMin") ; }
    else
     { hOverEPtMin_ = conf.getParameter<double>("hOverEPtMinPflow") ; }
   }
  else
   {
    hcalRecHits_ = conf.getParameter<edm::InputTag>("hcalRecHits") ;
    hOverEHBMinE_ = conf.getParameter<double>("hOverEHBMinE") ;
    hOverEHFMinE_ = conf.getParameter<double>("hOverEHFMinE") ;
   }
 }

void ElectronHcalHelper::checkSetup( const edm::EventSetup & es )
 {
  if (!useTowers_)
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
  if (useTowers_)
   {
    delete towerIso1_ ; towerIso1_ = 0 ;
    delete towerIso2_ ; towerIso2_ = 0 ;
    delete towersH_ ; towersH_ = 0 ;

    towersH_ = new edm::Handle<CaloTowerCollection>() ;
    if (!evt.getByLabel(hcalTowers_,*towersH_))
     { edm::LogError("ElectronHcalHelper::readEvent")<<"failed to get the hcal towers of label "<<hcalTowers_ ; }
    towerIso1_ = new EgammaTowerIsolation(hOverEConeSize_,0.,hOverEPtMin_,1,towersH_->product()) ;
    towerIso2_ = new EgammaTowerIsolation(hOverEConeSize_,0.,hOverEPtMin_,2,towersH_->product()) ;
   }
  else
   {
    delete hcalIso_ ; hcalIso_ = 0 ;
    delete mhbhe_ ; mhbhe_ = 0 ;
    delete hbhe_ ; hbhe_ = 0 ;

    hbhe_=  new edm::Handle<HBHERecHitCollection>() ;
    if (!evt.getByLabel(hcalRecHits_,*hbhe_))
     { edm::LogError("ElectronHcalHelper::readEvent")<<"failed to get the rechits of label "<<hcalRecHits_ ; }
    mhbhe_=  new HBHERecHitMetaCollection(**hbhe_) ;
    hcalIso_ = new EgammaHcalIsolation(hOverEConeSize_,0.,hOverEHBMinE_,hOverEHFMinE_,0.,0.,caloGeom_,mhbhe_) ;
   }
 }

double ElectronHcalHelper::hcalESum( const SuperCluster & sc )
 {
  if (useTowers_)
   { return(hcalESumDepth1(sc)+hcalESumDepth2(sc)) ; }
  else
   { return hcalIso_->getHcalESum(&sc) ; }
 }

double ElectronHcalHelper::hcalESumDepth1( const SuperCluster & sc )
 {
  if (useTowers_)
   { return towerIso1_->getTowerESum(&sc) ; }
  else
   { return hcalIso_->getHcalESumDepth1(&sc) ; }
 }

double ElectronHcalHelper::hcalESumDepth2( const SuperCluster & sc )
 {
  if (useTowers_)
   { return towerIso2_->getTowerESum(&sc) ; }
  else
   { return hcalIso_->getHcalESumDepth2(&sc) ; }
 }

ElectronHcalHelper::~ElectronHcalHelper()
 {
  if (useTowers_)
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


