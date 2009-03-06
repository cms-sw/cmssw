// Producer for making electron seeds from Si Strip hits.

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/SiStripElectronSeedGenerator.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "SiStripElectronSeedProducer.h"

#include <iostream>

using namespace reco;

SiStripElectronSeedProducer::SiStripElectronSeedProducer(const edm::ParameterSet& iConfig) : conf_(iConfig)
{
  matcher_ = new SiStripElectronSeedGenerator();

  //  get labels from config
  superClusters_[0]=iConfig.getParameter<edm::InputTag>("barrelSuperClusters");
  superClusters_[1]=iConfig.getParameter<edm::InputTag>("endcapSuperClusters");

  //register your products
  produces<ElectronSeedCollection>();
}

SiStripElectronSeedProducer::~SiStripElectronSeedProducer()
{
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  delete matcher_;
}

void SiStripElectronSeedProducer::produce(edm::Event& e, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace std;

  LogDebug("entering");
  LogDebug("")  <<"[SiStripElectronSeedProducer::produce] entering " ;

  matcher_->setupES(iSetup);

  ElectronSeedCollection *seeds = new ElectronSeedCollection;
  std::auto_ptr<ElectronSeedCollection> pSeeds;

  // do both barrel and endcap instances
  for (unsigned int i=0; i<2; i++) {

    // get the superclusters
    edm::Handle<SuperClusterCollection> clusters;
    if(e.getByLabel(superClusters_[i],clusters)) {
      // run the seed generator and put the ElectronSeeds into a collection
      matcher_->run(e,iSetup,clusters,*seeds);
    }

  }

  pSeeds = std::auto_ptr<ElectronSeedCollection>(seeds);

  e.put(pSeeds);

}

