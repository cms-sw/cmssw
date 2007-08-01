/** \file LaserHitPairGeneratorFromLayerPair.cc
 *  
 *
 *  $Date: 2007/05/10 12:00:46 $
 *  $Revision: 1.14 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignment/interface/LaserHitPairGeneratorFromLayerPair.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "Alignment/LaserAlignment/interface/OrderedLaserHitPairs.h"
#include "RecoTracker/TkTrackingRegions/interface/HitRZCompatibility.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionBase.h"

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

void LaserHitPairGeneratorFromLayerPair::hitPairs(const TrackingRegion & region, OrderedLaserHitPairs & result, const edm::EventSetup & iSetup)
{
	typedef OrderedLaserHitPair::InnerHit InnerHit;
	typedef OrderedLaserHitPair::OuterHit OuterHit;

	if (theInnerLayer->recHits().empty()) return;
	if (theOuterLayer->recHits().empty()) return;

	std::vector<OrderedLaserHitPair> allthepairs;

	std::vector<const TrackingRecHit*>::const_iterator ohh;

	for(ohh=theOuterLayer->recHits().begin();ohh!=theOuterLayer->recHits().end();ohh++){
    GlobalPoint oh = trackerGeometry->idToDet((*ohh)->geographicalId())->surface().toGlobal((*ohh)->localPosition());
	  std::vector<const TrackingRecHit*>::const_iterator ihh;
	  for(ihh=theInnerLayer->recHits().begin();ihh!=theInnerLayer->recHits().end();ihh++){
      GlobalPoint ih = trackerGeometry->idToDet((*ihh)->geographicalId())->surface().toGlobal((*ihh)->localPosition());

      double inny = ih.y();
      double outy = oh.y();
			double innz = ih.z();
			double outz = oh.z();
			double innphi = ih.phi();
			double outphi = oh.phi();
			double phi_diff = innphi - outphi;
      double r_diff = sqrt(pow(ih.x(),2)+pow(ih.y(),2)) - sqrt(pow(oh.x(),2)+pow(oh.y(),2));

			if ( ( inny * outy > 0.0 ) && ( innz * outz > 0.0 ) && ( fabs(phi_diff) < 0.005 ) && ( fabs(r_diff) < 0.5 ) )
			{
				allthepairs.push_back( OrderedLaserHitPair(*ihh, *ohh ));
			}
		}
	}
	stable_sort(allthepairs.begin(),allthepairs.end(),CompareHitPairsZ(iSetup));

	if (allthepairs.size() > 0) 
	{
		for (std::vector<OrderedLaserHitPair>::const_iterator it = allthepairs.begin(); it != allthepairs.end(); it++)
		{
			result.push_back(*it);
		}
	}
}
