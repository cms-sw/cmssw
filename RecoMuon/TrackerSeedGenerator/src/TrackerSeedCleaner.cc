/*
 * \class TrackerSeedCleaner
 *  Seeds Cleaner based on direction
 *  $Date: 2008/02/28 22:17:48 $
 *  $Revision: 1.0 $
    \author A. Grelli -  Purdue University, Pavia University
 */

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/TrackerSeedGenerator/plugins/TSGFromL2Muon.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateClosestToBeamLineBuilder.h"
#include "TrackingTools/TrajectoryState/interface/PerigeeConversions.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h" 
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/TkTrackingRegionsMargin.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGenerator.h"
#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGeneratorFactory.h"
#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedCleaner.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include <vector>

using namespace std;
using namespace edm;

// whatever you need for inizialization
void TrackerSeedCleaner::init(const MuonServiceProxy *service){

  theProxyService = service;
}
// Event Setup
void TrackerSeedCleaner::setEvent(const edm::Event& event)
{
  theEvent = &event;
}

// the cleaner
std::vector<TrajectorySeed > TrackerSeedCleaner::clean( const reco::TrackRef& muR, const RectangularEtaPhiTrackingRegion& region, tkSeeds& seeds ) {
 
 const edm::EventSetup & es = theProxyService->eventSetup();

 edm::ESHandle<MagneticField> theMF;
 es.get<IdealMagneticFieldRecord>().get(theMF);
 es.get<TransientRecHitRecord>().get(builderName,theTTRHBuilder);
 
 LogDebug("TrackerSeedCleaner")<<seeds.size()<<" trajectory seeds to the events before cleaning"<<endl;

 //retrieve beam spot information
 edm::Handle<reco::BeamSpot> bsHandle;
 theEvent->getByLabel(theBeamSpotTag, bsHandle);
 //cechk the validity otherwise vertexing
 const reco::BeamSpot & bs = *bsHandle;

 /*reco track and seeds as argoments. Seeds eta and phi are checked and 
   based on deviation from L2 eta and phi seed is accepted or not*/  

 std::vector<TrajectorySeed > result;

 TrajectoryStateTransform tsTransform;
 TrajectoryStateClosestToBeamLineBuilder tscblBuilder;
 // PerigeeConversions tspConverter;

 for(TrajectorySeedCollection::iterator seed = seeds.begin(); seed<seeds.end(); ++seed){

	//get parameters and errors from the seed state
	TransientTrackingRecHit::RecHitPointer recHit = theTTRHBuilder->build(&*(seed->recHits().second-1));
	TrajectoryStateOnSurface state = tsTransform.transientState( seed->startingState(), recHit->surface(), theMF.product());

	TrajectoryStateClosestToBeamLine tsAtClosestApproachSeed = tscblBuilder(*state.freeState(),bs);//as in TrackProducerAlgorithms

	GlobalPoint vSeed1 = tsAtClosestApproachSeed.trackStateAtPCA().position();
	GlobalVector pSeed = tsAtClosestApproachSeed.trackStateAtPCA().momentum();
	GlobalPoint vSeed(vSeed1.x()-bs.x0(),vSeed1.y()-bs.y0(),vSeed1.z()-bs.z0());

        //eta,phi info from seeds 
	double etaSeed = state.globalMomentum().eta();
	double phiSeed = pSeed.phi(); 

        //if the limits are too stringent rescale limits
	typedef PixelRecoRange< float > Range;
	typedef TkTrackingRegionsMargin< float > Margin;

	Range etaRange = region.etaRange();
	double etaLimit = (fabs(fabs(etaRange.max()) - fabs(etaRange.mean())) <0.05) ? 0.05 : fabs(fabs(etaRange.max()) - fabs(etaRange.mean())) ;

	Margin phiMargin = region.phiMargin();
	double phiLimit = (phiMargin.right() < 0.05 ) ? 0.05 : phiMargin.right(); 

        // Clean  
	bool inEtaRange = etaSeed >= (etaRange.mean() - etaLimit) && etaSeed <= (etaRange.mean() + etaLimit) ;
	
	bool inPhiRange = (fabs(deltaPhi(phiSeed,double(region.direction().phi()))) < phiLimit ) ? true : false ;

        if( inEtaRange && inPhiRange ) {
	  LogDebug("trackerSeedCleaner")<<"Keeping the seed";
	  result.push_back(*seed);
	} else {
	  seed = seeds.erase(seed);
	}
	
        LogDebug("TrackerSeedCleaner")<<" eta for current seed "<<etaSeed<<"\n"
                                      <<" phi for current seed "<<phiSeed<<"\n"
                                      <<" eta for L2 track  "<<muR->eta()<<"\n"
                                      <<" phi for L2 track  "<<muR->phi()<<"\n";
  }

  LogDebug("TrackerSeedCleaner")<<result.size()<<" trajectory seeds to the events afther cleaning"<<endl;

  return result;

}

