#include "FastSimulation/Muons/plugins/FastTSGFromIOHit.h"

/** \class FastTSGFromIOHit
 *
 *  Emulate TSGFromIOHit in RecoMuon
 *
 *  \author Adam Everett - Purdue University 
 */

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2DCollection.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeed.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "DataFormats/Math/interface/deltaPhi.h"


FastTSGFromIOHit::FastTSGFromIOHit(const edm::ParameterSet & iConfig,edm::ConsumesCollector& iC) : theConfig (iConfig)
{
  theCategory = "FastSimulation|Muons||FastTSGFromIOHit";

  thePtCut = iConfig.getParameter<double>("PtCut");

  theSeedCollectionLabels = iConfig.getParameter<std::vector<edm::InputTag> >("SeedCollectionLabels");

  theSimTrackCollectionLabel  = iConfig.getParameter<edm::InputTag>("SimTrackCollectionLabel");

}

FastTSGFromIOHit::~FastTSGFromIOHit()
{

  LogTrace(theCategory) << " FastTSGFromIOHit dtor called ";

}

void FastTSGFromIOHit::trackerSeeds(const TrackCand& staMuon, const TrackingRegion& region, std::vector<TrajectorySeed> & result) {
  
  // Retrieve the Monte Carlo truth (SimTracks)
  edm::Handle<edm::SimTrackContainer> theSimTracks;
  getEvent()->getByLabel(theSimTrackCollectionLabel,theSimTracks);
    
  // Retrieve Seed collection
  unsigned seedCollections = theSeedCollectionLabels.size();
  std::vector<edm::Handle<edm::View<TrajectorySeed> > > theSeeds;
  theSeeds.resize(seedCollections);
  unsigned seed_size = 0;
  for ( unsigned iseed=0; iseed<seedCollections; ++iseed ) { 
    getEvent()->getByLabel(theSeedCollectionLabels[iseed], theSeeds[iseed]);
    seed_size += theSeeds[iseed]->size();
  }
  
  // Make a ref to l2 muon
  reco::TrackRef muRef(staMuon.second);
  
  // Cut on muons with low momenta
  if ( muRef->pt() < thePtCut 
       || muRef->innerMomentum().Rho() < thePtCut 
       || muRef->innerMomentum().R() < 2.5 ){
  }return;
  
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
       
      const RectangularEtaPhiTrackingRegion & regionRef = dynamic_cast<const RectangularEtaPhiTrackingRegion & > (region);
  
      if( clean(muRef,regionRef,aSeed,theSimTrack) ) tkSeeds.push_back(*aSeed);
      tkIds.insert(simTrackId);
      
    } // End loop on seeds
 
  } // End loop on seed collections
  
  // Now create the Muon Trajectory Seed
  unsigned int is=0;
  unsigned int isMax=tkSeeds.size();
  for (;is!=isMax;++is){
    result.push_back( L3MuonTrajectorySeed(tkSeeds[is], muRef));
  } // End of tk seed loop
  
}

bool
FastTSGFromIOHit::clean(reco::TrackRef muRef,
			const RectangularEtaPhiTrackingRegion& region,
			const BasicTrajectorySeed* aSeed,
			const SimTrack& theSimTrack) 
{
  //  return true; 
  //}

  // Eta cleaner   
  const PixelRecoRange<float>& etaRange = region.etaRange() ;  
  //  return true;
  //}

  double etaSeed = theSimTrack.momentum().Eta();
  double etaLimit  = (fabs(fabs(etaRange.max())-fabs(etaRange.mean())) <0.05) ? 
    0.05 : fabs(fabs(etaRange.max()) - fabs(etaRange.mean())) ;
  bool inEtaRange = 
    etaSeed >= (etaRange.mean() - etaLimit) && 
    etaSeed <= (etaRange.mean() + etaLimit) ;
  if  ( !inEtaRange ) return false;
  
  // Phi cleaner
  const TkTrackingRegionsMargin<float>& phiMargin = region.phiMargin();
  double phiSeed = theSimTrack.momentum().Phi(); 
  double phiLimit  = (phiMargin.right() < 0.05 ) ? 0.05 : phiMargin.right(); 
  bool inPhiRange = 
    (fabs(deltaPhi(phiSeed,double(region.direction().phi()))) < phiLimit );
  if  ( !inPhiRange ) return false;
  
  // pt cleaner
  double ptSeed  = std::sqrt(theSimTrack.momentum().Perp2());
  double ptMin   = (region.ptMin()>3.5) ? 3.5: region.ptMin();  
  bool inPtRange = ptSeed >= ptMin &&  ptSeed<= 2*(muRef->pt());
  return inPtRange;
  
}
