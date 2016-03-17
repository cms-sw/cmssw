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
#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHit.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeed.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "DataFormats/Math/interface/deltaPhi.h"

FastTSGFromIOHit::FastTSGFromIOHit(const edm::ParameterSet & iConfig,edm::ConsumesCollector& iC) 
{

  theCategory = "FastSimulation|Muons||FastTSGFromIOHit";

  thePtCut = iConfig.getParameter<double>("PtCut");

  simTracksTk = iC.consumes<edm::SimTrackContainer>(iConfig.getParameter<edm::InputTag>("SimTrackCollectionLabel"));
  const auto & seedLabels = iConfig.getParameter<std::vector<edm::InputTag> >("SeedCollectionLabels");
  for(const auto & seedLabel : seedLabels)
  {
      seedsTks.push_back(iC.consumes<TrajectorySeedCollection>(seedLabel));
  }

}

FastTSGFromIOHit::~FastTSGFromIOHit()
{

  LogTrace(theCategory) << " FastTSGFromIOHit dtor called ";

}

void FastTSGFromIOHit::trackerSeeds(const TrackCand& staMuon, const TrackingRegion& region, const TrackerTopology *tTopo, std::vector<TrajectorySeed> & result) 
{
    // Make a ref to l2 muon
    reco::TrackRef muRef(staMuon.second);

    // Cut on muons with low momenta
    if ( muRef->pt() < thePtCut 
	 || muRef->innerMomentum().Rho() < thePtCut 
	 || muRef->innerMomentum().R() < 2.5 )
    {
	return;
    }
    
    // Retrieve the Monte Carlo truth (SimTracks)
    edm::Handle<edm::SimTrackContainer> simTracks;
    getEvent()->getByToken(simTracksTk,simTracks);
    
    // Retrieve Seed collection
    std::vector<edm::Handle<TrajectorySeedCollection> > seedCollections;
    seedCollections.resize(seedsTks.size());
    for ( unsigned iSeed = 0 ; iSeed < seedsTks.size() ; iSeed++) 
    {
	getEvent()->getByToken(seedsTks[iSeed],seedCollections[iSeed]);
    }
    
    // cast the tracking region
    const RectangularEtaPhiTrackingRegion & regionRef = dynamic_cast<const RectangularEtaPhiTrackingRegion & > (region);
  
    // select and store seeds
    std::set<unsigned> simTrackIds;
    for ( const auto & seeds : seedCollections) 
    { 
	for ( const auto & seed : *seeds)
	{
	    // Find the simtrack corresponding to the seed
	    TrajectorySeed::range recHitRange = seed.recHits();
	    const FastTrackerRecHit * firstRecHit = (const FastTrackerRecHit*) (&(*(recHitRange.first)));
	    int simTrackId = firstRecHit->simTrackId(0);
	    const SimTrack & simTrack = (*simTracks)[simTrackId];
      
	    // skip if simTrack already associated to a seed
	    if( simTrackIds.find(simTrackId) != simTrackIds.end() )
	    {
		continue;	
	    }
	    simTrackIds.insert(simTrackId);
      
	    // filter seed
	    if( !clean(muRef,regionRef,&seed,simTrack) )
	    {
		continue;
	    }

	    // store results
	    result.push_back(L3MuonTrajectorySeed(seed,muRef));
	} 
    } 
}

bool
FastTSGFromIOHit::clean(reco::TrackRef muRef,
			const RectangularEtaPhiTrackingRegion& region,
			const TrajectorySeed* aSeed,
			const SimTrack& theSimTrack) 
{

  // Eta cleaner   
  const PixelRecoRange<float>& etaRange = region.etaRange() ;  

  double etaSeed = theSimTrack.momentum().Eta();
  double etaLimit  = (fabs(fabs(etaRange.max())-fabs(etaRange.mean())) <0.05) ? 
    0.05 : fabs(fabs(etaRange.max()) - fabs(etaRange.mean())) ;
  bool inEtaRange = 
    etaSeed >= (etaRange.mean() - etaLimit) && 
    etaSeed <= (etaRange.mean() + etaLimit) ;
  if  ( !inEtaRange )
  {
      return false;
  }
  
  // Phi cleaner
  const TkTrackingRegionsMargin<float>& phiMargin = region.phiMargin();
  double phiSeed = theSimTrack.momentum().Phi(); 
  double phiLimit  = (phiMargin.right() < 0.05 ) ? 0.05 : phiMargin.right(); 
  bool inPhiRange = 
    (fabs(deltaPhi(phiSeed,double(region.direction().phi()))) < phiLimit );
  if  ( !inPhiRange )
  {
      return false;
  }
  
  // pt cleaner
  double ptSeed  = std::sqrt(theSimTrack.momentum().Perp2());
  double ptMin   = (region.ptMin()>3.5) ? 3.5: region.ptMin();  
  bool inPtRange = ptSeed >= ptMin &&  ptSeed<= 2*(muRef->pt());
  if  ( !inPtRange )
  {
      return false;
  }
  return true;
  
}
