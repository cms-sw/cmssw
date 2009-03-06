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
// $Id: ElectronSeedProducer.cc,v 1.2 2009/02/05 11:39:44 chamont Exp $
//
//

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronSeedGenerator.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/SeedFilter.h"

#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"

#include "ElectronSeedProducer.h"

#include <string>

using namespace reco;

ElectronSeedProducer::ElectronSeedProducer(const edm::ParameterSet& iConfig) :conf_(iConfig),seedFilter_(0),cacheID_(0)
{
  edm::ParameterSet pset = iConfig.getParameter<edm::ParameterSet>("SeedConfiguration");
  initialSeeds_=pset.getParameter<edm::InputTag>("initialSeeds");
  SCEtCut_=pset.getParameter<double>("SCEtCut");
  maxHOverE_=pset.getParameter<double>("maxHOverE");
  fromTrackerSeeds_=pset.getParameter<bool>("fromTrackerSeeds");
  prefilteredSeeds_=pset.getParameter<bool>("preFilteredSeeds");

  matcher_ = new ElectronSeedGenerator(pset);

  if (prefilteredSeeds_) seedFilter_ = new SeedFilter(pset);

  //  get collections from config'
  superClusters_[0]=iConfig.getParameter<edm::InputTag>("barrelSuperClusters");
  superClusters_[1]=iConfig.getParameter<edm::InputTag>("endcapSuperClusters");
  hcalRecHits_ = pset.getParameter<edm::InputTag>("hcalRecHits");

  //register your products
  produces<ElectronSeedCollection>();
}


ElectronSeedProducer::~ElectronSeedProducer()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
      delete matcher_;
      delete seedFilter_;
}

void ElectronSeedProducer::produce(edm::Event& e, const edm::EventSetup& iSetup)
{
  LogDebug("ElectronSeedProducer");
  LogDebug("ElectronSeedProducer")  <<"[ElectronSeedProducer::produce] entering " ;

  // get calo geometry
  if (cacheID_!=iSetup.get<CaloGeometryRecord>().cacheIdentifier()) {
    iSetup.get<CaloGeometryRecord>().get(theCaloGeom);
    cacheID_=iSetup.get<CaloGeometryRecord>().cacheIdentifier();
  }

  matcher_->setupES(iSetup);

  // get Hcal Rechit collection
  edm::Handle<HBHERecHitCollection> hbhe;
  HBHERecHitMetaCollection *mhbhe=0;
  bool got =    e.getByLabel(hcalRecHits_,hbhe);
  if (got) mhbhe=  new HBHERecHitMetaCollection(*hbhe);

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

  calc_=HoECalculator(theCaloGeom);

  // loop over barrel + endcap
  for (unsigned int i=0; i<2; i++) {
   // invoke algorithm
    edm::Handle<SuperClusterCollection> clusters;
    if (e.getByLabel(superClusters_[i],clusters))   {
	SuperClusterRefVector clusterRefs;
	filterClusters(clusters,mhbhe,clusterRefs);
	if ((fromTrackerSeeds_) && (prefilteredSeeds_)) filterSeeds(e,iSetup,clusterRefs);
        matcher_->run(e,iSetup,clusterRefs,theInitialSeedColl,*seeds);

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

void ElectronSeedProducer::filterClusters(const edm::Handle<reco::SuperClusterCollection> &superClusters,
    HBHERecHitMetaCollection*mhbhe, SuperClusterRefVector &sclRefs) {

  // filter the superclusters
  // - with EtCut
  // - with HoE using hcal rechit behind supercluster position
  for (unsigned int i=0;i<superClusters->size();++i) {
    const SuperCluster &scl=(*superClusters)[i];

    if (scl.energy()/cosh(scl.eta())>SCEtCut_) {

      double HoE=calc_(&scl,mhbhe,2);
      if (HoE <= maxHOverE_) {
	sclRefs.push_back(edm::Ref<reco::SuperClusterCollection> (superClusters,i));
      }

    }

  }
  LogDebug("ElectronSeedProducer")  <<"Filtered out "<<sclRefs.size() <<" superclusters from "<<superClusters->size() ;
}

void ElectronSeedProducer::filterSeeds(edm::Event& e, const edm::EventSetup& setup, reco::SuperClusterRefVector &sclRefs)
{

  for  (unsigned int i=0;i<sclRefs.size();++i) {
    seedFilter_->seeds(e, setup, sclRefs[i], theInitialSeedColl);

    LogDebug("ElectronSeedProducer")<< "Number fo Seeds: " << theInitialSeedColl->size() ;
  }


}
