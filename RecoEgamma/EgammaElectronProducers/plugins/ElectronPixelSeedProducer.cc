// -*- C++ -*-
//
// Package:    ElectronProducers
// Class:      ElectronPixelSeedProducer
// 
/**\class ElectronPixelSeedProducer RecoEgamma/ElectronProducers/src/ElectronPixelSeedProducer.cc

 Description: EDProducer of ElectronPixelSeed objects

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
// $Id: ElectronPixelSeedProducer.cc,v 1.22 2008/04/21 09:50:47 uberthon Exp $
//
//

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronPixelSeedGenerator.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/SeedFilter.h"

#include "DataFormats/EgammaReco/interface/ElectronPixelSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronPixelSeedFwd.h"

#include "ElectronPixelSeedProducer.h"

#include <string>

using namespace reco;
 
ElectronPixelSeedProducer::ElectronPixelSeedProducer(const edm::ParameterSet& iConfig) :conf_(iConfig),seedFilter_(0),cacheID_(0)
{
  edm::ParameterSet pset = iConfig.getParameter<edm::ParameterSet>("SeedConfiguration");
  initialSeeds_=pset.getParameter<edm::InputTag>("initialSeeds");
  SCEtCut_=pset.getParameter<double>("SCEtCut");
  maxHOverE_=pset.getParameter<double>("maxHOverE");
  fromTrackerSeeds_=pset.getParameter<bool>("fromTrackerSeeds");
  prefilteredSeeds_=pset.getParameter<bool>("preFilteredSeeds");

  matcher_ = new ElectronPixelSeedGenerator(pset);
 
  if (prefilteredSeeds_) seedFilter_ = new SeedFilter(pset);

  //  get collections from config'
  superClusters_[0]=iConfig.getParameter<edm::InputTag>("barrelSuperClusters");
  superClusters_[1]=iConfig.getParameter<edm::InputTag>("endcapSuperClusters");
  hcalRecHits_ = pset.getParameter<edm::InputTag>("hcalRecHits");

  //register your products
  produces<ElectronPixelSeedCollection>();
}


ElectronPixelSeedProducer::~ElectronPixelSeedProducer()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
      delete matcher_;
      delete seedFilter_;
}

void ElectronPixelSeedProducer::beginJob(edm::EventSetup const&iSetup) 
{
}

void ElectronPixelSeedProducer::produce(edm::Event& e, const edm::EventSetup& iSetup) 
{
  LogDebug("ElectronPixelSeedProducer");
  LogDebug("ElectronPixelSeedProducer")  <<"[ElectronPixelSeedProducer::produce] entering " ;
  // get calo geometry 
  if (cacheID_!=iSetup.get<IdealGeometryRecord>().cacheIdentifier()) {
              iSetup.get<IdealGeometryRecord>().get(theCaloGeom);
	      cacheID_=iSetup.get<IdealGeometryRecord>().cacheIdentifier();
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
 


  ElectronPixelSeedCollection *seeds= new ElectronPixelSeedCollection;
std::auto_ptr<ElectronPixelSeedCollection> pSeeds;

  // loop over barrel + endcap
  calc_=HoECalculator(theCaloGeom);
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
  pSeeds=  std::auto_ptr<ElectronPixelSeedCollection>(seeds);
  for (ElectronPixelSeedCollection::iterator is=pSeeds->begin(); is!=pSeeds->end();is++) {
    LogDebug("ElectronPixelSeedProducer")  << "new seed with " << (*is).nHits() << " hits, charge " << (*is).getCharge() <<
	" and cluster energy " << (*is).superCluster()->energy() << " PID "<<(*is).superCluster().id();
  }
  e.put(pSeeds);
  delete mhbhe;
  if (fromTrackerSeeds_ && prefilteredSeeds_) delete theInitialSeedColl;
 }

void ElectronPixelSeedProducer::filterClusters(const edm::Handle<reco::SuperClusterCollection> &superClusters,HBHERecHitMetaCollection*mhbhe, SuperClusterRefVector &sclRefs) {

  // filter the superclusters
  // - with EtCut
  // with HoE
  for (unsigned int i=0;i<superClusters->size();++i) {
    const SuperCluster &scl=(*superClusters)[i];

    if (scl.energy()/cosh(scl.eta())>SCEtCut_) {

      double HoE=calc_(&(*scl.seed()),mhbhe);
      if (HoE <= maxHOverE_) {
	sclRefs.push_back(edm::Ref<reco::SuperClusterCollection> (superClusters,i));
      }
    }
  }
  LogDebug("ElectronPixelSeedProducer")  <<"Filtered out "<<sclRefs.size() <<" superclusters from "<<superClusters->size() ;
}

void ElectronPixelSeedProducer::filterSeeds(edm::Event& e, const edm::EventSetup& setup, reco::SuperClusterRefVector &sclRefs)
{

  for  (unsigned int i=0;i<sclRefs.size();++i) {
    seedFilter_->seeds(e, setup, sclRefs[i], theInitialSeedColl);

    LogDebug("ElectronPixelSeedProducer")<< "Number fo Seeds: " << theInitialSeedColl->size() ;
  }
 

}
