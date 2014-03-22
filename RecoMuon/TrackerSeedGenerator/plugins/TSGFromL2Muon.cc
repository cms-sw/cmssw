#include "TSGFromL2Muon.h"

TSGFromL2Muon::TSGFromL2Muon(const edm::ParameterSet& cfg)
  : theConfig(cfg), theService(0), theRegionBuilder(0), theTkSeedGenerator(0), theSeedCleaner(0)
{
  produces<L3MuonTrajectorySeedCollection>();

  edm::ParameterSet serviceParameters = cfg.getParameter<edm::ParameterSet>("ServiceParameters");
  theService = new MuonServiceProxy(serviceParameters);

  thePtCut = cfg.getParameter<double>("PtCut");
  thePCut = cfg.getParameter<double>("PCut");

  theL2CollectionLabel = cfg.getParameter<edm::InputTag>("MuonCollectionLabel");

  edm::ConsumesCollector iC  = consumesCollector();

  //region builder
  edm::ParameterSet regionBuilderPSet = theConfig.getParameter<edm::ParameterSet>("MuonTrackingRegionBuilder");
  //ability to no define a region
  if (!regionBuilderPSet.empty()){
    theRegionBuilder = new MuonTrackingRegionBuilder(regionBuilderPSet, iC);
  }

  //seed generator
  //std::string seedGenPSetLabel = theConfig.getParameter<std::string>("tkSeedGenerator");
  //edm::ParameterSet seedGenPSet = theConfig.getParameter<edm::ParameterSet>(seedGenPSetLabel);
  edm::ParameterSet seedGenPSet = theConfig.getParameter<edm::ParameterSet>("TkSeedGenerator");
  std::string seedGenName = seedGenPSet.getParameter<std::string>("ComponentName");

  theTkSeedGenerator = TrackerSeedGeneratorFactory::get()->create(seedGenName, seedGenPSet,iC);  
  
  //seed cleaner
  edm::ParameterSet trackerSeedCleanerPSet = theConfig.getParameter<edm::ParameterSet>("TrackerSeedCleaner");
  //to activate or not the cleaner
  if (!trackerSeedCleanerPSet.empty()){
    theSeedCleaner = new TrackerSeedCleaner(trackerSeedCleanerPSet,iC);
  }


  l2muonToken = consumes<reco::TrackCollection>(theL2CollectionLabel);
}

TSGFromL2Muon::~TSGFromL2Muon()
{
  delete theService;
  if (theSeedCleaner) delete theSeedCleaner;
  delete theTkSeedGenerator;
  if (theRegionBuilder) delete theRegionBuilder;
}

void TSGFromL2Muon::beginRun(const edm::Run & run, const edm::EventSetup&es)
{
  //update muon proxy service
  theService->update(es);
  
  if (theRegionBuilder) theRegionBuilder->init(theService);
  theTkSeedGenerator->init(theService);
  if (theSeedCleaner) theSeedCleaner->init(theService);

}

void TSGFromL2Muon::produce(edm::Event& ev, const edm::EventSetup& es)
{
  std::auto_ptr<L3MuonTrajectorySeedCollection> result(new L3MuonTrajectorySeedCollection());

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHand;
  es.get<IdealGeometryRecord>().get(tTopoHand);
  const TrackerTopology *tTopo=tTopoHand.product();


  //intialize tools
  theService->update(es);
  theTkSeedGenerator->setEvent(ev);
  if (theRegionBuilder)  theRegionBuilder->setEvent(ev);
  if (theSeedCleaner) theSeedCleaner->setEvent(ev);

  //retrieve L2 track collection
  edm::Handle<reco::TrackCollection> l2muonH;
  ev.getByToken(l2muonToken ,l2muonH); 

  // produce trajectoryseed collection
  unsigned int imu=0;
  unsigned int imuMax=l2muonH->size();
  LogDebug("TSGFromL2Muon")<<imuMax<<" l2 tracks.";

  for (;imu!=imuMax;++imu){
    //make a ref to l2 muon
    reco::TrackRef muRef(l2muonH, imu);
    
    // cut on muons with low momenta
    if ( muRef->pt() < thePtCut 
	 || muRef->innerMomentum().Rho() < thePtCut 
	 || muRef->innerMomentum().R() < thePCut ) continue;
    
    //define the region of interest
    std::unique_ptr<RectangularEtaPhiTrackingRegion> region;
    if(theRegionBuilder){
      region.reset(theRegionBuilder->region(muRef));
    }
    
    //get the seeds
    std::vector<TrajectorySeed> tkSeeds;
    //make this stupid TrackCand
    std::pair<const Trajectory*,reco::TrackRef> staCand((Trajectory*)(0), muRef);
    theTkSeedGenerator->trackerSeeds(staCand, *region, tTopo,tkSeeds);

    //Seed Cleaner From Direction
    //clean them internatly
    if(theSeedCleaner){
       theSeedCleaner->clean(muRef,*region,tkSeeds);
       LogDebug("TSGFromL2Muon") << tkSeeds.size() << " seeds for this L2 afther cleaning.";
    }

    unsigned int is=0;
    unsigned int isMax=tkSeeds.size();
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

