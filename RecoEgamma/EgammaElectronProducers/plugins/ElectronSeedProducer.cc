// -*- C++ -*-
//
// Package:    ElectronProducers
// Class:      ElectronSeedProducer
//
/**\class ElectronSeedProducer RecoEgamma/ElectronProducers/src/ElectronSeedProducer.cc

 Description: EDProducer of ElectronSeed objects

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
// $Id: ElectronSeedProducer.cc,v 1.23 2011/01/14 21:25:58 chamont Exp $
//
//

#include "ElectronSeedProducer.h"

#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHcalIsolation.h"
//#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronSeedGenerator.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronHcalHelper.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/SeedFilter.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronUtilities.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "RecoCaloTools/Selectors/interface/CaloConeSelector.h"

#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

using namespace reco ;

ElectronSeedProducer::ElectronSeedProducer( const edm::ParameterSet& iConfig )
 : beamSpotTag_("offlineBeamSpot"),
   //conf_(iConfig),
   seedFilter_(0), applyHOverECut_(true), hcalHelper_(0),
   caloGeom_(0), caloGeomCacheId_(0), caloTopo_(0), caloTopoCacheId_(0)
 {
  conf_ = iConfig.getParameter<edm::ParameterSet>("SeedConfiguration") ;

  initialSeeds_ = conf_.getParameter<edm::InputTag>("initialSeeds") ;
  SCEtCut_ = conf_.getParameter<double>("SCEtCut") ;
  fromTrackerSeeds_ = conf_.getParameter<bool>("fromTrackerSeeds") ;
  prefilteredSeeds_ = conf_.getParameter<bool>("preFilteredSeeds") ;

  // new beamSpot tag
  if (conf_.exists("beamSpot"))
   { beamSpotTag_ = conf_.getParameter<edm::InputTag>("beamSpot") ; }

  // for H/E
//  if (conf_.exists("applyHOverECut"))
//   { applyHOverECut_ = conf_.getParameter<bool>("applyHOverECut") ; }
  applyHOverECut_ = conf_.getParameter<bool>("applyHOverECut") ;
  if (applyHOverECut_)
   {
    ElectronHcalHelper::Configuration hcalCfg ;
    hcalCfg.hOverEConeSize = conf_.getParameter<double>("hOverEConeSize") ;
    if (hcalCfg.hOverEConeSize>0)
     {
      hcalCfg.useTowers = true ;
      hcalCfg.hcalTowers = conf_.getParameter<edm::InputTag>("hcalTowers") ;
      hcalCfg.hOverEPtMin = conf_.getParameter<double>("hOverEPtMin") ;
     }
    hcalHelper_ = new ElectronHcalHelper(hcalCfg) ;
    maxHOverEBarrel_=conf_.getParameter<double>("maxHOverEBarrel") ;
    maxHOverEEndcaps_=conf_.getParameter<double>("maxHOverEEndcaps") ;
    maxHBarrel_=conf_.getParameter<double>("maxHBarrel") ;
    maxHEndcaps_=conf_.getParameter<double>("maxHEndcaps") ;
//    hOverEConeSize_=conf_.getParameter<double>("hOverEConeSize") ;
//    hOverEHBMinE_=conf_.getParameter<double>("hOverEHBMinE") ;
//    hOverEHFMinE_=conf_.getParameter<double>("hOverEHFMinE") ;
   }

  matcher_ = new ElectronSeedGenerator(conf_) ;

  //  get collections from config'
  superClusters_[0]=iConfig.getParameter<edm::InputTag>("barrelSuperClusters") ;
  superClusters_[1]=iConfig.getParameter<edm::InputTag>("endcapSuperClusters") ;

  //register your products
  produces<ElectronSeedCollection>() ;
}


void ElectronSeedProducer::beginRun(edm::Run const&, edm::EventSetup const&)
 {
  // FIXME: because of a bug presumably in tracker seeding,
  // perhaps in CombinedHitPairGenerator, badly caching some EventSetup product,
  // we must redo the SeedFilter for each run.
  if (prefilteredSeeds_) seedFilter_ = new SeedFilter(conf_) ;
 }

void ElectronSeedProducer::endRun(edm::Run const&, edm::EventSetup const&)
 {
  delete seedFilter_ ;
  seedFilter_ = 0 ;
 }

ElectronSeedProducer::~ElectronSeedProducer()
 {
  delete hcalHelper_ ;
  delete matcher_ ;
 }

void ElectronSeedProducer::produce(edm::Event& e, const edm::EventSetup& iSetup)
 {
  LogDebug("ElectronSeedProducer") <<"[ElectronSeedProducer::produce] entering " ;

  edm::Handle<reco::BeamSpot> theBeamSpot ;
  e.getByLabel(beamSpotTag_,theBeamSpot) ;

  if (hcalHelper_)
   {
    hcalHelper_->checkSetup(iSetup) ;
    hcalHelper_->readEvent(e) ;
   }

  // get calo geometry
  if (caloGeomCacheId_!=iSetup.get<CaloGeometryRecord>().cacheIdentifier()) {
    iSetup.get<CaloGeometryRecord>().get(caloGeom_);
    caloGeomCacheId_=iSetup.get<CaloGeometryRecord>().cacheIdentifier();
  }
  if (caloTopoCacheId_!=iSetup.get<CaloTopologyRecord>().cacheIdentifier()){
    caloTopoCacheId_=iSetup.get<CaloTopologyRecord>().cacheIdentifier();
    iSetup.get<CaloTopologyRecord>().get(caloTopo_);
  }

  matcher_->setupES(iSetup);

  // get initial TrajectorySeeds if necessary
  if (fromTrackerSeeds_)
   {
    if (!prefilteredSeeds_)
     {
      edm::Handle<TrajectorySeedCollection> hSeeds;
      e.getByLabel(initialSeeds_, hSeeds);
      theInitialSeedColl = const_cast<TrajectorySeedCollection *> (hSeeds.product());
     }
    else
     { theInitialSeedColl = new TrajectorySeedCollection ; }
   }
  else
   { theInitialSeedColl = 0 ; } // not needed in this case

  ElectronSeedCollection * seeds = new ElectronSeedCollection ;

  // loop over barrel + endcap
  for (unsigned int i=0; i<2; i++)
   {
    edm::Handle<SuperClusterCollection> clusters ;
    if (e.getByLabel(superClusters_[i],clusters))
     {
      SuperClusterRefVector clusterRefs ;
      std::vector<float> hoe1s, hoe2s ;
      filterClusters(*theBeamSpot,clusters,/*mhbhe_,*/clusterRefs,hoe1s,hoe2s) ;
      if ((fromTrackerSeeds_) && (prefilteredSeeds_))
       { filterSeeds(e,iSetup,clusterRefs) ; }
      matcher_->run(e,iSetup,clusterRefs,hoe1s,hoe2s,theInitialSeedColl,*seeds) ;
     }
   }

  // store the accumulated result
  std::auto_ptr<ElectronSeedCollection> pSeeds(seeds) ;
  ElectronSeedCollection::iterator is ;
  for ( is=pSeeds->begin() ; is!=pSeeds->end() ; is++ )
   {
    edm::RefToBase<CaloCluster> caloCluster = is->caloCluster() ;
    SuperClusterRef superCluster = caloCluster.castTo<SuperClusterRef>() ;
    LogDebug("ElectronSeedProducer")<< "new seed with "
      << (*is).nHits() << " hits"
      << ", charge " << (*is).getCharge()
      << " and cluster energy " << superCluster->energy()
      << " PID "<<superCluster.id() ;
   }
  e.put(pSeeds) ;
  if (fromTrackerSeeds_ && prefilteredSeeds_) delete theInitialSeedColl;
 }


//===============================
// Filter the superclusters
// - with EtCut
// - with HoE using calo cone
//===============================

void ElectronSeedProducer::filterClusters
 ( const reco::BeamSpot & bs,
   const edm::Handle<reco::SuperClusterCollection> & superClusters,
   /*HBHERecHitMetaCollection * mhbhe,*/ SuperClusterRefVector & sclRefs,
   std::vector<float> & hoe1s, std::vector<float> & hoe2s )
 {
  for (unsigned int i=0;i<superClusters->size();++i)
   {
    const SuperCluster & scl = (*superClusters)[i] ;
    double sclEta = EleRelPoint(scl.position(),bs.position()).eta() ;
    if (scl.energy()/cosh(sclEta)>SCEtCut_)
     {
//      if ((applyHOverECut_==true)&&((hcalHelper_->hcalESum(scl)/scl.energy()) > maxHOverE_))
//       { continue ; }
//      sclRefs.push_back(edm::Ref<reco::SuperClusterCollection>(superClusters,i)) ;
       double had1, had2, had, scle ;
       bool HoeVeto = false ;
       if (applyHOverECut_==true)
        {
         had1 = hcalHelper_->hcalESumDepth1(scl);
         had2 = hcalHelper_->hcalESumDepth2(scl);
         had = had1+had2 ;
         scle = scl.energy() ;
         int detector = scl.seed()->hitsAndFractions()[0].first.subdetId() ;
         if (detector==EcalBarrel && (had<maxHBarrel_ || had/scle<maxHOverEBarrel_)) HoeVeto=true;
         else if (detector==EcalEndcap && (had<maxHEndcaps_ || had/scle<maxHOverEEndcaps_)) HoeVeto=true;
         if (HoeVeto)
          {
           sclRefs.push_back(edm::Ref<reco::SuperClusterCollection>(superClusters,i)) ;
           hoe1s.push_back(had1/scle) ;
           hoe2s.push_back(had2/scle) ;
          }
        }
       else
        {
         sclRefs.push_back(edm::Ref<reco::SuperClusterCollection>(superClusters,i)) ;
         hoe1s.push_back(std::numeric_limits<float>::infinity()) ;
         hoe2s.push_back(std::numeric_limits<float>::infinity()) ;
        }
     }
   }
  LogDebug("ElectronSeedProducer")<<"Filtered out "<<sclRefs.size()<<" superclusters from "<<superClusters->size() ;
 }

void ElectronSeedProducer::filterSeeds
 ( edm::Event & event, const edm::EventSetup & setup,
   reco::SuperClusterRefVector & sclRefs )
 {
  for ( unsigned int i=0 ; i<sclRefs.size() ; ++i )
   {
    seedFilter_->seeds(event,setup,sclRefs[i],theInitialSeedColl) ;
    LogDebug("ElectronSeedProducer")<<"Number of Seeds: "<<theInitialSeedColl->size() ;
   }
 }
