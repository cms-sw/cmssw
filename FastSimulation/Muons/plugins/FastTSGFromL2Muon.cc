#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeedCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2DCollection.h" 
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/GlobalTrackingTools/interface/MuonTrackingRegionBuilder.h"

#include "FastSimulation/Muons/plugins/FastTSGFromL2Muon.h"

//#include <TH1.h>

#include <set>

FastTSGFromL2Muon::FastTSGFromL2Muon(const edm::ParameterSet& cfg) : theConfig(cfg) 
{
  produces<L3MuonTrajectorySeedCollection>();

  edm::ParameterSet serviceParameters = 
    cfg.getParameter<edm::ParameterSet>("ServiceParameters");
  theService = new MuonServiceProxy(serviceParameters);

  thePtCut = cfg.getParameter<double>("PtCut");

  theL2CollectionLabel = cfg.getParameter<edm::InputTag>("MuonCollectionLabel");
  theSeedCollectionLabels = cfg.getParameter<std::vector<edm::InputTag> >("SeedCollectionLabels");
  theSimTrackCollectionLabel  = cfg.getParameter<edm::InputTag>("SimTrackCollectionLabel");
  // useTFileService_ = cfg.getUntrackedParameter<bool>("UseTFileService",false);

}

FastTSGFromL2Muon::~FastTSGFromL2Muon()
{
}

void 
FastTSGFromL2Muon::beginRun(edm::Run const& run, edm::EventSetup const& es)
{
  //update muon proxy service
  theService->update(es);
  
  //region builder
  edm::ParameterSet regionBuilderPSet = 
    theConfig.getParameter<edm::ParameterSet>("MuonTrackingRegionBuilder");
  edm::ConsumesCollector iC  = consumesCollector();
  theRegionBuilder = new MuonTrackingRegionBuilder(regionBuilderPSet,theService,iC);
  
  /*
  if(useTFileService_) {
    edm::Service<TFileService> fs;
    h_nSeedPerTrack = fs->make<TH1F>("nSeedPerTrack","nSeedPerTrack",76,-0.5,75.5);
    h_nGoodSeedPerTrack = fs->make<TH1F>("nGoodSeedPerTrack","nGoodSeedPerTrack",75,-0.5,75.5);
    h_nGoodSeedPerEvent = fs->make<TH1F>("nGoodSeedPerEvent","nGoodSeedPerEvent",75,-0.5,75.5);
  } else {
    h_nSeedPerTrack = 0;
    h_nGoodSeedPerEvent = 0;
    h_nGoodSeedPerTrack = 0;
  }
  */

}


void 
FastTSGFromL2Muon::produce(edm::Event& ev, const edm::EventSetup& es)
{

  // Initialize the output product
  std::auto_ptr<L3MuonTrajectorySeedCollection> result(new L3MuonTrajectorySeedCollection());

  //intialize service
  theService->update(es);
  
  // Region builder
  theRegionBuilder->setEvent(ev);

  // Retrieve the Monte Carlo truth (SimTracks)
  edm::Handle<edm::SimTrackContainer> theSimTracks;
  ev.getByLabel(theSimTrackCollectionLabel,theSimTracks);

  // Retrieve L2 muon collection
  edm::Handle<reco::TrackCollection> l2muonH;
  ev.getByLabel(theL2CollectionLabel ,l2muonH); 

  // Retrieve Seed collection
  unsigned seedCollections = theSeedCollectionLabels.size();
  std::vector<edm::Handle<edm::View<TrajectorySeed> > > theSeeds;
  theSeeds.resize(seedCollections);
  unsigned seed_size = 0;
  for ( unsigned iseed=0; iseed<seedCollections; ++iseed ) { 
    ev.getByLabel(theSeedCollectionLabels[iseed], theSeeds[iseed]);
    seed_size += theSeeds[iseed]->size();
  }

  // Loop on L2 muons
  unsigned int imu=0;
  unsigned int imuMax=l2muonH->size();
  // std::cout << "Found " << imuMax << " L2 muons" << std::endl;
  for (;imu!=imuMax;++imu){

    // Make a ref to l2 muon
    reco::TrackRef muRef(l2muonH, imu);
    
    // Cut on muons with low momenta
    if ( muRef->pt() < thePtCut 
	 || muRef->innerMomentum().Rho() < thePtCut 
	 || muRef->innerMomentum().R() < 2.5 ) continue;
    
    // Define the region of interest
    RectangularEtaPhiTrackingRegion * region = theRegionBuilder->region(muRef);

    // Copy the collection of seeds (ahem, this is time consuming!)
    std::vector<TrajectorySeed> tkSeeds;
    std::set<unsigned> tkIds;
    tkSeeds.reserve(seed_size);
    for ( unsigned iseed=0; iseed<seedCollections; ++iseed ) { 
      edm::Handle<edm::View<TrajectorySeed> > aSeedCollection = theSeeds[iseed];
      unsigned nSeeds = aSeedCollection->size();
      for (unsigned seednr = 0; seednr < nSeeds; ++seednr) {
            
	// The seed
	const BasicTrajectorySeed* aSeed = &((*aSeedCollection)[seednr]);
	
	// Find the first hit of the Seed
	TrajectorySeed::range theSeedingRecHitRange = aSeed->recHits();
	const SiTrackerGSMatchedRecHit2D * theFirstSeedingRecHit = 
	  (const SiTrackerGSMatchedRecHit2D*) (&(*(theSeedingRecHitRange.first)));

	// The SimTrack id associated to that recHit
	int simTrackId = theFirstSeedingRecHit->simtrackId();

	// Track already associated to a seed
	std::set<unsigned>::iterator tkId = tkIds.find(simTrackId);
	if( tkId != tkIds.end() ) continue;	

	const SimTrack& theSimTrack = (*theSimTracks)[simTrackId]; 

	if ( clean(muRef,region,aSeed,theSimTrack) ) tkSeeds.push_back(*aSeed);
	tkIds.insert(simTrackId);

      } // End loop on seeds
 
    } // End loop on seed collections
  
    // Free memory
    delete region;

    // A plot
    // if(h_nSeedPerTrack) h_nSeedPerTrack->Fill(tkSeeds.size());

    // Another plot
    // if(h_nGoodSeedPerTrack) h_nGoodSeedPerTrack->Fill(tkSeeds.size());
    

    // Now create the Muon Trajectory Seed
    unsigned int is=0;
    unsigned int isMax=tkSeeds.size();
    for (;is!=isMax;++is){
      result->push_back( L3MuonTrajectorySeed(tkSeeds[is], muRef));
    } // End of tk seed loop
    
  } // End of l2 muon loop

  // std::cout << "Found " << result->size() << " seeds for muons" << std::endl;

  // And yet another plot
  // if(h_nGoodSeedPerEvent) h_nGoodSeedPerEvent->Fill(result->size());
  
  //put in the event
  ev.put(result);

}

bool
FastTSGFromL2Muon::clean(reco::TrackRef muRef,
			 RectangularEtaPhiTrackingRegion* region,
			 const BasicTrajectorySeed* aSeed, 
			 const SimTrack& theSimTrack) { 
    
  // Eta cleaner   
  const PixelRecoRange<float>& etaRange = region->etaRange() ;  
  double etaSeed = theSimTrack.momentum().Eta();
  double etaLimit  = (fabs(fabs(etaRange.max())-fabs(etaRange.mean())) <0.05) ? 
    0.05 : fabs(fabs(etaRange.max()) - fabs(etaRange.mean())) ;
  bool inEtaRange = 
    etaSeed >= (etaRange.mean() - etaLimit) && 
    etaSeed <= (etaRange.mean() + etaLimit) ;
  if  ( !inEtaRange ) return false;
  
  // Phi cleaner
  const TkTrackingRegionsMargin<float>& phiMargin = region->phiMargin();
  double phiSeed = theSimTrack.momentum().Phi(); 
  double phiLimit  = (phiMargin.right() < 0.05 ) ? 0.05 : phiMargin.right(); 
  bool inPhiRange = 
    (fabs(deltaPhi(phiSeed,double(region->direction().phi()))) < phiLimit );
  if  ( !inPhiRange ) return false;
  
  // pt cleaner
  double ptSeed  = std::sqrt(theSimTrack.momentum().Perp2());
  double ptMin   = (region->ptMin()>3.5) ? 3.5: region->ptMin();  
  bool inPtRange = ptSeed >= ptMin &&  ptSeed<= 2*(muRef->pt());
  return inPtRange;
  
}
