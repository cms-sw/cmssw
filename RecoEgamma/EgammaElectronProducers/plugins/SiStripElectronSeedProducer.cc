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
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"

#include "SiStripElectronSeedProducer.h"

#include <iostream>

using namespace reco;

SiStripElectronSeedProducer::SiStripElectronSeedProducer(const edm::ParameterSet& iConfig)
{
  if(iConfig.exists("SeedConfiguration")){
    conf_ = iConfig.getParameter<edm::ParameterSet>("SeedConfiguration") ;
  }else{
    conf_.addParameter("beamSpot",edm::InputTag("offlineBeamSpot"));
    conf_.addParameter("tibOriginZCut",20.);
    conf_.addParameter("tidOriginZCut",20.);
    conf_.addParameter("tecOriginZCut",20.);
    conf_.addParameter("monoOriginZCut",20.);
    conf_.addParameter("tibDeltaPsiCut",0.1);
    conf_.addParameter("tidDeltaPsiCut",0.1);
    conf_.addParameter("tecDeltaPsiCut",0.1);
    conf_.addParameter("monoDeltaPsiCut",0.1);
    conf_.addParameter("tibPhiMissHit2Cut",0.006);
    conf_.addParameter("tidPhiMissHit2Cut",0.006);
    conf_.addParameter("tecPhiMissHit2Cut",0.007);
    conf_.addParameter("monoPhiMissHit2Cut",0.02);
    conf_.addParameter("tibZMissHit2Cut",0.35);
    conf_.addParameter("tidRMissHit2Cut",0.3);
    conf_.addParameter("tecRMissHit2Cut",0.3);
    conf_.addParameter("tidEtaUsage", 1.2);
    conf_.addParameter("tidMaxHits",4);
    conf_.addParameter("tecMaxHits",2);
    conf_.addParameter("monoMaxHits",4);
    conf_.addParameter("maxSeeds",5);
  }
  SiStripElectronSeedGenerator::Tokens ssesg_tokens;
  ssesg_tokens.token_bs = consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"));
  if (conf_.existsAs<edm::InputTag>("measurementTrackerEvent")) {
    ssesg_tokens.token_mte = consumes<MeasurementTrackerEvent>(conf_.getParameter<edm::InputTag>("measurementTrackerEvent"));
  }
  matcher_ = new SiStripElectronSeedGenerator(conf_,ssesg_tokens);

  

  //  get labels from config
  superClusters_[0]=consumes<reco::SuperClusterCollection>(iConfig.getParameter<edm::InputTag>("barrelSuperClusters"));
  superClusters_[1]=consumes<reco::SuperClusterCollection>(iConfig.getParameter<edm::InputTag>("endcapSuperClusters"));

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
    if(e.getByToken(superClusters_[i],clusters)) {
      // run the seed generator and put the ElectronSeeds into a collection
      matcher_->run(e,iSetup,clusters,*seeds);
    }

  }

  pSeeds = std::auto_ptr<ElectronSeedCollection>(seeds);

  e.put(pSeeds);

}

