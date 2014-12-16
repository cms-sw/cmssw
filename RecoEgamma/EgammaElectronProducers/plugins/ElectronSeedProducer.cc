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
// $Id: ElectronSeedProducer.cc,v 1.24 2013/02/28 08:35:10 eulisse Exp $
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

#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"

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
   seedFilter_(0), applyHOverECut_(true), hcalHelperBarrel_(0), hcalHelperEndcap_(0), 
   caloGeom_(0), caloGeomCacheId_(0), caloTopo_(0), caloTopoCacheId_(0)
 {
  conf_ = iConfig.getParameter<edm::ParameterSet>("SeedConfiguration") ;

  initialSeeds_ = conf_.getParameter<edm::InputTag>("initialSeeds") ;
  //SCEtCut_ = conf_.getParameter<double>("SCEtCut") ;
  SCEtCutBarrel_ = conf_.getParameter<double>("SCEtCutBarrel") ;
  SCEtCutEndcap_ = conf_.getParameter<double>("SCEtCutEndcap") ;
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
    ElectronHcalHelper::Configuration hcalCfgBarrel ;
    hcalCfgBarrel.hOverEConeSize = conf_.getParameter<double>("hOverEConeSize") ;
    //hcalCfg.hOverEMethod = conf_.getParameter<int>("hOverEMethod") ;
    hcalCfgBarrel.hOverEMethod = conf_.getParameter<int>("hOverEMethodBarrel") ;
    if (hcalCfgBarrel.hOverEConeSize>0)
     {
      hcalCfgBarrel.useTowers = true ;
      hcalCfgBarrel.hcalTowers = conf_.getParameter<edm::InputTag>("hcalTowers") ;
      //here the HCAL clusters
      if (hcalCfgBarrel.hOverEMethod==3)
       { 
        hcalCfgBarrel.hcalClusters = conf_.getParameter<edm::InputTag>("barrelHCALClusters") ;
       }
     }
    hcalCfgBarrel.hOverEPtMin = conf_.getParameter<double>("hOverEPtMin") ;
    hcalHelperBarrel_ = new ElectronHcalHelper(hcalCfgBarrel) ;

    ElectronHcalHelper::Configuration hcalCfgEndcap ;
    hcalCfgEndcap.hOverEConeSize = conf_.getParameter<double>("hOverEConeSize") ;
    //hcalCfg.hOverEMethod = conf_.getParameter<int>("hOverEMethod") ;
    hcalCfgEndcap.hOverEMethod = conf_.getParameter<int>("hOverEMethodEndcap") ;
    if (hcalCfgEndcap.hOverEConeSize>0)
     {
      hcalCfgEndcap.useTowers = true ;
      hcalCfgEndcap.hcalTowers = conf_.getParameter<edm::InputTag>("hcalTowers") ;
      //here the HCAL clusters
      if (hcalCfgEndcap.hOverEMethod==3)
       { 
        hcalCfgEndcap.hcalClusters = conf_.getParameter<edm::InputTag>("endcapHCALClusters") ;
       }
     }
    hcalCfgEndcap.hOverEPtMin = conf_.getParameter<double>("hOverEPtMin") ;
    hcalHelperEndcap_ = new ElectronHcalHelper(hcalCfgEndcap) ;

    maxHOverEBarrel_=conf_.getParameter<double>("maxHOverEBarrel") ;
    maxHOverEEndcaps_=conf_.getParameter<double>("maxHOverEEndcaps") ;
    maxHOverEOuterEndcaps_=conf_.getParameter<double>("maxHOverEOuterEndcaps") ;
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
  delete hcalHelperBarrel_ ;
  delete hcalHelperEndcap_ ;
  delete matcher_ ;
 }

void ElectronSeedProducer::produce(edm::Event& e, const edm::EventSetup& iSetup)
 {
  LogDebug("ElectronSeedProducer") <<"[ElectronSeedProducer::produce] entering " ;

  edm::Handle<reco::BeamSpot> theBeamSpot ;
  e.getByLabel(beamSpotTag_,theBeamSpot) ;

  if (hcalHelperBarrel_)
   {
    hcalHelperBarrel_->checkSetup(iSetup) ;
    hcalHelperBarrel_->readEvent(e) ;
   }

  if (hcalHelperEndcap_)
   {
    hcalHelperEndcap_->checkSetup(iSetup) ;
    hcalHelperEndcap_->readEvent(e) ;
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
    float sclPt = scl.energy()/cosh(sclEta);
    int detector = scl.seed()->hitsAndFractions()[0].first.subdetId() ;
    //if (scl.energy()/cosh(sclEta)>SCEtCut_)
    if ((sclPt > SCEtCutBarrel_ && detector == EcalBarrel) || (sclPt > SCEtCutEndcap_ && (detector == EcalEndcap || detector==EcalShashlik || detector == HGCEE) ))
      {
//      if ((applyHOverECut_==true)&&((hcalHelper_->hcalESum(scl)/scl.energy()) > maxHOverE_))
//       { continue ; }
//      sclRefs.push_back(edm::Ref<reco::SuperClusterCollection>(superClusters,i)) ;
	double had1(0), had2(0), had(0), scle(0) ;
       bool HoeVeto = false ;
       if (applyHOverECut_)
        {
	  if (detector==EcalBarrel) {
	    if( hcalHelperEndcap_->getConfig().hOverEMethod != 3 ) {
	      had1 = hcalHelperBarrel_->hcalESumDepth1(scl);
	      had2 = hcalHelperBarrel_->hcalESumDepth2(scl);
	    } else {
	      had1 = hcalHelperBarrel_->HCALClustersBehindSC(scl);
	    }
	  } else if (detector==EcalEndcap || detector==EcalShashlik) {
	    if( hcalHelperEndcap_->getConfig().hOverEMethod != 3 ) {
	      had1 = hcalHelperEndcap_->hcalESumDepth1(scl);
	      had2 = hcalHelperEndcap_->hcalESumDepth2(scl);
	    } else {
	      had1 = hcalHelperEndcap_->HCALClustersBehindSC(scl);
	    }
	  } else if (detector==EcalEndcap || detector==HGCEE) {
	    had1 = hcalHelperEndcap_->HCALClustersBehindSC(scl);
      }
         had = had1+had2 ;
         scle = scl.energy() ;
	     int component = scl.seed()->hitsAndFractions()[0].first.det() ;
         //int detector = scl.seed()->hitsAndFractions()[0].first.subdetId() ;
         if (component==DetId::Ecal && detector==EcalBarrel && (had<maxHBarrel_ || had/scle<maxHOverEBarrel_)) HoeVeto=true;
         else if (component==DetId::Ecal && (detector==EcalEndcap || detector==EcalShashlik ) && fabs(sclEta) < 2.65 && (had<maxHEndcaps_ || had/scle<maxHOverEEndcaps_)) HoeVeto=true;
         else if (component==DetId::Ecal && (detector==EcalEndcap || detector==EcalShashlik ) && fabs(sclEta) > 2.65 && (had<maxHEndcaps_ || had/scle<maxHOverEOuterEndcaps_)) HoeVeto=true;
	     else if (component==DetId::Forward && detector==HGCEE && (had<maxHEndcaps_ || had/scle<maxHOverEEndcaps_)) HoeVeto=true;
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
