#include "TSGFromL2Muon.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"

#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeedCollection.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/GlobalTrackingTools/interface/MuonTrackingRegionBuilder.h"
#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGenerator.h"
#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGeneratorFactory.h"
#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedCleaner.h"

TSGFromL2Muon::TSGFromL2Muon(const edm::ParameterSet& cfg)
  : theConfig(cfg), theTkSeedGenerator(0)
{
  produces<L3MuonTrajectorySeedCollection>();

  edm::ParameterSet serviceParameters = cfg.getParameter<edm::ParameterSet>("ServiceParameters");
  theService = new MuonServiceProxy(serviceParameters);

  thePtCut = cfg.getParameter<double>("PtCut");

  theL2CollectionLabel = cfg.getParameter<edm::InputTag>("MuonCollectionLabel");
}

TSGFromL2Muon::~TSGFromL2Muon()
{
}

void TSGFromL2Muon::beginJob(const edm::EventSetup& es)
{
  //update muon proxy service
  theService->update(es);
  
  //region builder
  edm::ParameterSet regionBuilderPSet = theConfig.getParameter<edm::ParameterSet>("MuonTrackingRegionBuilder");
  //ability to no define a region
  if (regionBuilderPSet.empty()){
    theRegionBuilder = 0;}
  else{
    theRegionBuilder = new MuonTrackingRegionBuilder(regionBuilderPSet,theService);
  }

  //seed generator
  std::string seedGenPSetLabel = theConfig.getParameter<std::string>("tkSeedGenerator");
  edm::ParameterSet seedGenPSet = theConfig.getParameter<edm::ParameterSet>(seedGenPSetLabel);
  std::string seedGenName = seedGenPSet.getParameter<std::string>("ComponentName");
  theTkSeedGenerator = TrackerSeedGeneratorFactory::get()->create(seedGenName, seedGenPSet);
  theTkSeedGenerator->init(theService);

  //seed cleaner
  edm::ParameterSet trackerSeedCleanerPSet = theConfig.getParameter<edm::ParameterSet>("TrackerSeedCleaner");
  //to activate or not the cleaner
  if (trackerSeedCleanerPSet.empty())
    theSeedCleaner=0;
  else{
    theSeedCleaner = new TrackerSeedCleaner(trackerSeedCleanerPSet);
    theSeedCleaner->init(theService);
  }

}


void TSGFromL2Muon::produce(edm::Event& ev, const edm::EventSetup& es)
{
  std::auto_ptr<L3MuonTrajectorySeedCollection> result(new L3MuonTrajectorySeedCollection());

  //intialize tools
  theService->update(es);
  theTkSeedGenerator->setEvent(ev);
  if (theRegionBuilder)  theRegionBuilder->setEvent(ev);
  if (theSeedCleaner) theSeedCleaner->setEvent(ev);

  //retrieve L2 track collection
  edm::Handle<reco::TrackCollection> l2muonH;
  ev.getByLabel(theL2CollectionLabel ,l2muonH); 

  // produce trajectoryseed collection
  uint imu=0;
  uint imuMax=l2muonH->size();
  LogDebug("TSGFromL2Muon")<<imuMax<<" l2 tracks.";

  for (;imu!=imuMax;++imu){
    //make a ref to l2 muon
    reco::TrackRef muRef(l2muonH, imu);
    
    // cut on muons with low momenta
    if ( muRef->pt() < thePtCut 
	 || muRef->innerMomentum().Rho() < thePtCut 
	 || muRef->innerMomentum().R() < 2.5 ) continue;
    
    //define the region of interest
    RectangularEtaPhiTrackingRegion region;
    if(theRegionBuilder){
      RectangularEtaPhiTrackingRegion * region1 = theRegionBuilder->region(muRef);
      
      TkTrackingRegionsMargin<float> etaMargin(fabs(region1->etaRange().min() - region1->etaRange().mean()),
					       fabs(region1->etaRange().max() - region1->etaRange().mean()));

      region=RectangularEtaPhiTrackingRegion(region1->direction(),
					     region1->origin(),
					     region1->ptMin(),
					     region1->originRBound(),
					     region1->originZBound(),
					     etaMargin,
					     region1->phiMargin());
      delete region1;
    }
    
    //get the seeds
    std::vector<TrajectorySeed> tkSeeds;
    //make this stupid TrackCand
    std::pair<const Trajectory*,reco::TrackRef> staCand(0, muRef);
    theTkSeedGenerator->trackerSeeds(staCand, region, tkSeeds);

    //Seed Cleaner From Direction
    //clean them internatly
    if(theSeedCleaner){
       std::vector<TrajectorySeed> tkClSeeds = theSeedCleaner->clean(muRef,region,tkSeeds);
       edm::LogWarning("TSGFromL2Muon")<<tkClSeeds.size() << " or " << tkSeeds.size() << " seeds for this L2 afther cleaning.";
       //tkSeeds.swap(tkClSeeds);
    }

    uint is=0;
    uint isMax=tkSeeds.size();
    LogDebug("TSGFromL2Muon")<<isMax<<" seeds for this L2.";
    for (;is!=isMax;++is){
      result->push_back( L3MuonTrajectorySeed(tkSeeds[is], muRef));
    }//tkseed loop
    
  }//l2muon loop
  
  //ADDME
  //remove seed duplicate, keeping the ref to L2

  LogDebug("TSGFromL2Muon")<<result->size()<<" trajectory seeds to the events";

  //put in the event
  ev.put(result);
}

