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
// $Id: ElectronPixelSeedProducer.cc,v 1.18 2008/03/03 17:09:01 uberthon Exp $
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
#include "RecoEgamma/EgammaElectronAlgos/interface/SubSeedGenerator.h"

#include "DataFormats/EgammaReco/interface/ElectronPixelSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronPixelSeedFwd.h"

#include "ElectronPixelSeedProducer.h"

#include <string>

using namespace reco;
 
ElectronPixelSeedProducer::ElectronPixelSeedProducer(const edm::ParameterSet& iConfig) : conf_(iConfig),cacheID_(0)
{

  algo_ = iConfig.getParameter<std::string>("SeedAlgo");
  edm::ParameterSet pset = iConfig.getParameter<edm::ParameterSet>("SeedConfiguration");
  SCEtCut_=pset.getParameter<double>("SCEtCut");
  maxHOverE_=pset.getParameter<double>("maxHOverE");

  if (algo_=="FilteredSeed") 
    matcher_= new SubSeedGenerator(pset);
  else matcher_ = new ElectronPixelSeedGenerator(pset);
 
 //  get labels from config'
  label_[0]=iConfig.getParameter<std::string>("superClusterBarrelProducer");
  instanceName_[0]=iConfig.getParameter<std::string>("superClusterBarrelLabel");
  label_[1]=iConfig.getParameter<std::string>("superClusterEndcapProducer");
  instanceName_[1]=iConfig.getParameter<std::string>("superClusterEndcapLabel");
  hbheLabel_ = pset.getParameter<std::string>("hbheModule");
  hbheInstanceName_ = pset.getParameter<std::string>("hbheInstance");

  //register your products
  produces<ElectronPixelSeedCollection>();
}


ElectronPixelSeedProducer::~ElectronPixelSeedProducer()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
      delete matcher_;
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
  bool got =    e.getByLabel(hbheLabel_,hbheInstanceName_,hbhe);  
  if (got) mhbhe=  new HBHERecHitMetaCollection(*hbhe);

  ElectronPixelSeedCollection *seeds= new ElectronPixelSeedCollection;
  std::auto_ptr<ElectronPixelSeedCollection> pSeeds;

  // loop over barrel + endcap
  calc_=HoECalculator(theCaloGeom);
  for (unsigned int i=0; i<2; i++) {  
   // invoke algorithm
    edm::Handle<SuperClusterCollection> clusters;
    if (e.getByLabel(label_[i],instanceName_[i],clusters))   {
      if (algo_=="") {
	SuperClusterRefVector clusterRefs;
	filterClusters(clusters,mhbhe,clusterRefs);
	matcher_->run(e,iSetup,clusterRefs,*seeds);
      }
      else  matcher_->run(e,iSetup,clusters,*seeds);
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
 }

void ElectronPixelSeedProducer::filterClusters(const edm::Handle<reco::SuperClusterCollection> &superClusters,HBHERecHitMetaCollection*mhbhe, SuperClusterRefVector &sclRefs) {

  // filter the superclusters
  // - with EtCut
  // with HoE
  for (unsigned int i=0;i<superClusters->size();++i) {
    const SuperCluster &scl=(*superClusters)[i];

    if (scl.energy()/cosh(scl.eta())>SCEtCut_) {

      double HoE=calc_(&scl,mhbhe);
      if (HoE <= maxHOverE_) {
	sclRefs.push_back(edm::Ref<reco::SuperClusterCollection> (superClusters,i));
      }
    }
  }
  LogDebug("ElectronPixelSeedProducer")  <<"Filtered out "<<sclRefs.size() <<" superclusters from "<<superClusters->size() ;
}
