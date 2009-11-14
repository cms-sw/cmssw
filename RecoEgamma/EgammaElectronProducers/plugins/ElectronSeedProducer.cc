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
// $Id: ElectronSeedProducer.cc,v 1.9 2009/10/18 21:42:13 chamont Exp $
//
//

#include "ElectronSeedProducer.h"

#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHcalIsolation.h"
//#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronSeedGenerator.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronHcalHelper.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/SeedFilter.h"

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
 :
   //conf_(iConfig),
   seedFilter_(0), applyHOverECut_(true), hcalHelper_(0)
   , caloGeom_(0), caloGeomCacheId_(0), caloTopo_(0), caloTopoCacheId_(0)
 {
  edm::ParameterSet pset = iConfig.getParameter<edm::ParameterSet>("SeedConfiguration") ;

  initialSeeds_ = pset.getParameter<edm::InputTag>("initialSeeds") ;
  SCEtCut_ = pset.getParameter<double>("SCEtCut") ;
  fromTrackerSeeds_ = pset.getParameter<bool>("fromTrackerSeeds") ;
  prefilteredSeeds_ = pset.getParameter<bool>("preFilteredSeeds") ;

  // for H/E
  if (pset.exists("applyHOverECut"))
   { applyHOverECut_ = pset.getParameter<bool>("applyHOverECut") ; }
  if (applyHOverECut_)
   {
    hcalHelper_ = new ElectronHcalHelper(pset) ;
    maxHOverEBarrel_=pset.getParameter<double>("maxHOverEBarrel") ;
    maxHOverEEndcaps_=pset.getParameter<double>("maxHOverEEndcaps") ;
    maxHBarrel_=pset.getParameter<double>("maxHBarrel") ;
    maxHEndcaps_=pset.getParameter<double>("maxHEndcaps") ;
//    hOverEConeSize_=pset.getParameter<double>("hOverEConeSize") ;
//    hOverEHBMinE_=pset.getParameter<double>("hOverEHBMinE") ;
//    hOverEHFMinE_=pset.getParameter<double>("hOverEHFMinE") ;
   }

  matcher_ = new ElectronSeedGenerator(pset) ;

  if (prefilteredSeeds_) seedFilter_ = new SeedFilter(pset) ;

  //  get collections from config'
  superClusters_[0]=iConfig.getParameter<edm::InputTag>("barrelSuperClusters") ;
  superClusters_[1]=iConfig.getParameter<edm::InputTag>("endcapSuperClusters") ;

  //register your products
  produces<ElectronSeedCollection>() ;
}


ElectronSeedProducer::~ElectronSeedProducer()
 {
  delete hcalHelper_ ;
  delete matcher_ ;
  delete seedFilter_ ;
 }

void ElectronSeedProducer::produce(edm::Event& e, const edm::EventSetup& iSetup)
 {
  LogDebug("ElectronSeedProducer") <<"[ElectronSeedProducer::produce] entering " ;

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
  if (fromTrackerSeeds_) {
    if (!prefilteredSeeds_) {
      edm::Handle<TrajectorySeedCollection> hSeeds;
      e.getByLabel(initialSeeds_, hSeeds);
      theInitialSeedColl = const_cast<TrajectorySeedCollection *> (hSeeds.product());
    }
    else theInitialSeedColl =new TrajectorySeedCollection;
  }else
    theInitialSeedColl=0;// not needed in this case

  ElectronSeedCollection *seeds= new ElectronSeedCollection;

  // loop over barrel + endcap
  for (unsigned int i=0; i<2; i++)
   {
    edm::Handle<SuperClusterCollection> clusters ;
    if (e.getByLabel(superClusters_[i],clusters))
     {
	  SuperClusterRefVector clusterRefs ;
	  filterClusters(clusters,/*mhbhe_,*/clusterRefs) ;
	  if ((fromTrackerSeeds_) && (prefilteredSeeds_))
	   { filterSeeds(e,iSetup,clusterRefs) ; }
      matcher_->run(e,iSetup,clusterRefs,theInitialSeedColl,*seeds) ;
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
 ( const edm::Handle<reco::SuperClusterCollection> & superClusters,
   /*HBHERecHitMetaCollection * mhbhe,*/ SuperClusterRefVector & sclRefs )
 {
  for (unsigned int i=0;i<superClusters->size();++i)
   {
    const SuperCluster & scl = (*superClusters)[i] ;
    if (scl.energy()/cosh(scl.eta())>SCEtCut_)
     {
//      if ((applyHOverECut_==true)&&((hcalHelper_->hcalESum(scl)/scl.energy()) > maxHOverE_))
//       { continue ; }
//      sclRefs.push_back(edm::Ref<reco::SuperClusterCollection>(superClusters,i)) ;
       double had;
       bool HoEveto = false;
       if (applyHOverECut_==true) {
         had=hcalHelper_->hcalESum(scl);
         int detector = scl.seed()->hitsAndFractions()[0].first.subdetId() ;
         if (detector==EcalBarrel && (had<maxHBarrel_ || had/scl.energy()<maxHOverEBarrel_)) HoEveto=true;
         else if (detector==EcalEndcap && (had<maxHEndcaps_ || had/scl.energy()<maxHOverEEndcaps_)) HoEveto=true;
       }	 
       if (!applyHOverECut_ || HoEveto) sclRefs.push_back(edm::Ref<reco::SuperClusterCollection>(superClusters,i)) ;
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
