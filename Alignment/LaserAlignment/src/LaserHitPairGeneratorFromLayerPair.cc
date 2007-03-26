/** \file LaserHitPairGeneratorFromLayerPair.cc
 *  
 *
 *  $Date: 2007/03/26 08:12:48 $
 *  $Revision: 1.6 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignment/interface/LaserHitPairGeneratorFromLayerPair.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkHitPairs/interface/OrderedHitPairs.h"
#include "RecoTracker/TkTrackingRegions/interface/HitRZCompatibility.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionBase.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapLoop.h"

#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

typedef ctfseeding::SeedingHit TkHitPairsCachedHit;

LaserHitPairGeneratorFromLayerPair::LaserHitPairGeneratorFromLayerPair(const LayerWithHits* inner, 
	const LayerWithHits* outer, const edm::EventSetup& iSetup) : trackerGeometry(0),
	theInnerLayer(inner), theOuterLayer(outer)
{

  edm::ESHandle<TrackerGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
  trackerGeometry = tracker.product();
}

void LaserHitPairGeneratorFromLayerPair::hitPairs(const TrackingRegion & region, OrderedHitPairs & result, const edm::EventSetup & iSetup)
{
	typedef OrderedHitPair::InnerHit InnerHit;
	typedef OrderedHitPair::OuterHit OuterHit;

	// const LayerHitMap & innerHitsMap = theLayerCache(theInnerLayer, region, iSetup);
// 	if (innerHitsMap.empty()) return;
	if (theInnerLayer->recHits().empty()) return;

	// const LayerHitMap & outerHitsMap = theLayerCache(theOuterLayer, region, iSetup);
// 	if (outerHitsMap.empty()) return;
	if (theOuterLayer->recHits().empty()) return;

	std::vector<OrderedHitPair> allthepairs;
// 	const TkHitPairsCachedHit * oh;
// 	LayerHitMapLoop outerHits = outerHitsMap.loop();

// 	while ( (oh = outerHits.getHit()) )
// 	{
// 		LayerHitMapLoop innerHits = innerHitsMap.loop();
// 		const TkHitPairsCachedHit * ih;
// 		while ( (ih = innerHits.getHit()) )
// 		{
	std::vector<const TrackingRecHit*>::const_iterator ohh;
	for(ohh=theOuterLayer->recHits().begin();ohh!=theOuterLayer->recHits().end();ohh++){
	  const TkHitPairsCachedHit * oh=new TkHitPairsCachedHit(*ohh,iSetup);
	  std::vector<const TrackingRecHit*>::const_iterator ihh;
	  for(ihh=theInnerLayer->recHits().begin();ihh!=theInnerLayer->recHits().end();ihh++){
	    const TkHitPairsCachedHit * ih=new TkHitPairsCachedHit(*ihh,iSetup);

			double inny = ih->r() * sin(ih->phi());
			double outy = oh->r() * sin(oh->phi());
			double innz = ih->z();
			double outz = oh->z();
			double innphi = ih->phi();
			double outphi = oh->phi();
			double phi_diff = innphi - outphi;
			double r_diff = ih->r() - oh->r();

			if ( ( inny * outy > 0.0 ) && ( innz * outz > 0.0 ) && ( fabs(phi_diff) < 0.005 ) && ( fabs(r_diff) < 0.5 ) )
			{
				allthepairs.push_back( OrderedHitPair(*ih, *oh ));
			}
		}
	}
	stable_sort(allthepairs.begin(),allthepairs.end(),CompareHitPairsZ(iSetup));

	if (allthepairs.size() > 0) 
	{
		for (std::vector<OrderedHitPair>::const_iterator it = allthepairs.begin(); it != allthepairs.end(); it++)
		{
			result.push_back(*it);
		}
	}
}
