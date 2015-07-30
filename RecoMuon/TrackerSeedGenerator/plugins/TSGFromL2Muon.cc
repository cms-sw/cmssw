#include "TSGFromL2Muon.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/GlobalTrackingTools/interface/MuonTrackingRegionBuilder.h"
#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGenerator.h"
#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGeneratorFactory.h"
#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedCleaner.h"

TSGFromL2Muon::TSGFromL2Muon(const edm::ParameterSet& cfg)
  : theConfig(cfg), theService(0), theRegionBuilder(0), theTkSeedGenerator(0), theSeedCleaner(0){
  produces<L3MuonTrajectorySeedCollection>();

  edm::ConsumesCollector iC  = consumesCollector();

  edm::ParameterSet serviceParameters = cfg.getParameter<edm::ParameterSet>("ServiceParameters");
  theService = new MuonServiceProxy(serviceParameters);

  //Pt and P cuts
  thePtCut = cfg.getParameter<double>("PtCut");
  thePCut = cfg.getParameter<double>("PCut");

  //Region builder
  edm::ParameterSet regionBuilderPSet = theConfig.getParameter<edm::ParameterSet>("MuonTrackingRegionBuilder");
  //ability to no define a region
  if (!regionBuilderPSet.empty()){
    theRegionBuilder = new MuonTrackingRegionBuilder(regionBuilderPSet, iC);
  }

  //Seed generator
  edm::ParameterSet seedGenPSet = theConfig.getParameter<edm::ParameterSet>("TkSeedGenerator");
  std::string seedGenName = seedGenPSet.getParameter<std::string>("ComponentName");
  theTkSeedGenerator = TrackerSeedGeneratorFactory::get()->create(seedGenName, seedGenPSet,iC);  
  
  //Seed cleaner
  edm::ParameterSet trackerSeedCleanerPSet = theConfig.getParameter<edm::ParameterSet>("TrackerSeedCleaner");
  //To activate or not the cleaner
  if (!trackerSeedCleanerPSet.empty()){
    theSeedCleaner = new TrackerSeedCleaner(trackerSeedCleanerPSet,iC);
  }

  //L2 collection
  theL2CollectionLabel = cfg.getParameter<edm::InputTag>("MuonCollectionLabel");
  l2muonToken = consumes<reco::TrackCollection>(theL2CollectionLabel);
}


TSGFromL2Muon::~TSGFromL2Muon(){
  delete theService;
  if (theSeedCleaner) delete theSeedCleaner;
  delete theTkSeedGenerator;
  if (theRegionBuilder) delete theRegionBuilder;
}


void TSGFromL2Muon::beginRun(const edm::Run & run, const edm::EventSetup&es){
  //update muon proxy service
  theService->update(es);
  theTkSeedGenerator->init(theService);
  if (theSeedCleaner) theSeedCleaner->init(theService);
}


void TSGFromL2Muon::produce(edm::Event& ev, const edm::EventSetup& es){
  std::auto_ptr<L3MuonTrajectorySeedCollection> result(new L3MuonTrajectorySeedCollection());

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHand;
  es.get<TrackerTopologyRcd>().get(tTopoHand);
  const TrackerTopology *tTopo=tTopoHand.product();

  //intialize tools
  theService->update(es);
  theTkSeedGenerator->setEvent(ev);
  if (theRegionBuilder)  theRegionBuilder->setEvent(ev);
  if (theSeedCleaner) theSeedCleaner->setEvent(ev);

  //retrieve L2 track collection
  edm::Handle<reco::TrackCollection> l2muonH;
  ev.getByToken(l2muonToken ,l2muonH); 

  // produce trajectory seed collection
  LogDebug("TSGFromL2Muon")<<l2muonH->size()<<" l2 tracks.";

  for (unsigned int imu=0; imu != l2muonH->size(); ++imu){
    //make a ref to l2 muon
    reco::TrackRef muRef(l2muonH, imu);
    
    // cut on muons with low momenta
    if ( muRef->pt() < thePtCut 
	 || muRef->innerMomentum().Rho() < thePtCut 
	 || muRef->innerMomentum().R() < thePCut ) continue;
    
    //define the region of interest
    std::unique_ptr<RectangularEtaPhiTrackingRegion> region;
    if(theRegionBuilder){
      region = theRegionBuilder->region(muRef);
    }
    
    //Make seeds container
    std::vector<TrajectorySeed> tkSeeds;

    //Make TrackCand
    std::pair<const Trajectory*,reco::TrackRef> staCand((Trajectory*)(0), muRef);

    //Run seed generator to fill seed container
    theTkSeedGenerator->trackerSeeds(staCand, *region, tTopo,tkSeeds);

    //Seed Cleaner From Direction
    if(theSeedCleaner){
       theSeedCleaner->clean(muRef,*region,tkSeeds);
    }

    for (unsigned int is=0; is != tkSeeds.size(); ++is){
      result->push_back( L3MuonTrajectorySeed(tkSeeds[is], muRef));
    }
    
  }
  

  //ADDME: remove seed duplicate, keeping the ref to L2

  LogDebug("TSGFromL2Muon")<<result->size()<<" trajectory seeds to the events";

  //put in the event
  ev.put(result);
}

// FillDescription generated from hltL3TrajSeedOIState with additions from OIHit and IOHit
void
TSGFromL2Muon::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setAllowAnything();
}

