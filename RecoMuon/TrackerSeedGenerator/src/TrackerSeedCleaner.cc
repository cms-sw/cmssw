/*
 * \class TrackerSeedCleaner
 *  Reference class for seeds cleaning
 *  Seeds Cleaner based on sharedHits cleaning, direction cleaning and pt cleaning
 *  $Date: 2011/12/22 21:08:03 $
 *  $Revision: 1.10 $
    \author A. Grelli -  Purdue University, Pavia University
 */

#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedCleaner.h"

//---------------
// C++ Headers --
//---------------
#include <vector>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/TkTrackingRegionsMargin.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGenerator.h"
#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGeneratorFactory.h"

using namespace std;
using namespace edm;

//
// inizialization
//
void TrackerSeedCleaner::init(const MuonServiceProxy *service){

  theProxyService = service;
  
  theRedundantCleaner = new RedundantSeedCleaner();
}

//
//
//
void TrackerSeedCleaner::setEvent(const edm::Event& event)
{
 event.getByLabel(theBeamSpotTag, bsHandle_);
}

//
// clean seeds
//
void TrackerSeedCleaner::clean( const reco::TrackRef& muR, const RectangularEtaPhiTrackingRegion& region, tkSeeds& seeds ) {


 // call the shared input cleaner
 if(cleanBySharedHits) theRedundantCleaner->define(seeds);

 theProxyService->eventSetup().get<TransientRecHitRecord>().get(builderName_,theTTRHBuilder);

 LogDebug("TrackerSeedCleaner")<<seeds.size()<<" trajectory seeds to the events before cleaning"<<endl; 

 //check the validity otherwise vertexing
 const reco::BeamSpot & bs = *bsHandle_;
 /*reco track and seeds as arguments. Seeds eta and phi are checked and 
   based on deviation from L2 eta and phi seed is accepted or not*/  

 std::vector<TrajectorySeed > result;

 
 TSCBLBuilderNoMaterial tscblBuilder;
 // PerigeeConversions tspConverter;
 for(TrajectorySeedCollection::iterator seed = seeds.begin(); seed<seeds.end(); ++seed){
        if(seed->nHits() < 2) continue; 
	//get parameters and errors from the seed state
	TransientTrackingRecHit::RecHitPointer recHit = theTTRHBuilder->build(&*(seed->recHits().second-1));
	TrajectoryStateOnSurface state = trajectoryStateTransform::transientState( seed->startingState(), recHit->surface(), theProxyService->magneticField().product());

	TrajectoryStateClosestToBeamLine tsAtClosestApproachSeed = tscblBuilder(*state.freeState(),bs);//as in TrackProducerAlgorithms
	if (!tsAtClosestApproachSeed.isValid()) continue;
	GlobalPoint vSeed1 = tsAtClosestApproachSeed.trackStateAtPCA().position();
	GlobalVector pSeed = tsAtClosestApproachSeed.trackStateAtPCA().momentum();
	GlobalPoint vSeed(vSeed1.x()-bs.x0(),vSeed1.y()-bs.y0(),vSeed1.z()-bs.z0());


        //eta,phi info from seeds 
	double etaSeed = state.globalMomentum().eta();
	double phiSeed = pSeed.phi(); 

        //if the limits are too stringent rescale limits
	typedef PixelRecoRange< float > Range;
	typedef TkTrackingRegionsMargin< float > Margin;

	Range etaRange   = region.etaRange();
	double etaLimit  = (fabs(fabs(etaRange.max()) - fabs(etaRange.mean())) <0.1) ? 0.1 : fabs(fabs(etaRange.max()) - fabs(etaRange.mean())) ;

	Margin phiMargin = region.phiMargin();
	double phiLimit  = (phiMargin.right() < 0.1 ) ? 0.1 : phiMargin.right(); 

        double ptSeed  = pSeed.perp();
        double ptMin   = (region.ptMin()>3.5) ? 3.5: region.ptMin();
        // Clean  
	bool inEtaRange = etaSeed >= (etaRange.mean() - etaLimit) && etaSeed <= (etaRange.mean() + etaLimit) ;
	bool inPhiRange = (fabs(deltaPhi(phiSeed,double(region.direction().phi()))) < phiLimit );
        // pt cleaner
        bool inPtRange = ptSeed >= ptMin &&  ptSeed<= 2*(muR->pt());
        
        // save efficiency don't clean triplets with pt cleaner 
        if(seed->nHits()==3) inPtRange = true;

        // use pt and angle cleaners
        if(inPtRange  && usePt_Cleaner && !useDirection_Cleaner) {

            result.push_back(*seed);
            LogDebug("TrackerSeedCleaner")<<" Keeping the seed : this seed passed pt selection";
        }
        
        // use only angle default option
        if( inEtaRange && inPhiRange && !usePt_Cleaner && useDirection_Cleaner) {

            result.push_back(*seed);
            LogDebug("TrackerSeedCleaner")<<" Keeping the seed : this seed passed direction selection";

        }

        // use all the cleaners
        if( inEtaRange && inPhiRange && inPtRange && usePt_Cleaner && useDirection_Cleaner) {

            result.push_back(*seed);
            LogDebug("TrackerSeedCleaner")<<" Keeping the seed : this seed passed direction and pt selection";

        }

	
        LogDebug("TrackerSeedCleaner")<<" eta for current seed "<<etaSeed<<"\n"
                                      <<" phi for current seed "<<phiSeed<<"\n"
                                      <<" eta for L2 track  "<<muR->eta()<<"\n"
                                      <<" phi for L2 track  "<<muR->phi()<<"\n";


  }

   //the new seeds collection
   if(result.size()!=0 && (useDirection_Cleaner || usePt_Cleaner)) seeds.swap(result);

   LogDebug("TrackerSeedCleaner")<<seeds.size()<<" trajectory seeds to the events after cleaning"<<endl;
 
   return;

}

