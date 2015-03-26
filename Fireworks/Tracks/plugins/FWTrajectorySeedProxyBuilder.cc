// -*- C++ -*-
//
// Package:     TrajectorySeeds
// Class  :     FWTrajectorySeedProxyBuilder
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Nov 25 14:42:13 EST 2008
//

// system include files
#include "TEveStraightLineSet.h"
#include "TEvePointSet.h"
#include "TEveLine.h"

// user include files
#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWMagField.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"
#include "Fireworks/Tracks/interface/estimate_field.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"


class FWTrajectorySeedProxyBuilder : public FWSimpleProxyBuilderTemplate<TrajectorySeed> {

public:
   FWTrajectorySeedProxyBuilder();
   virtual ~FWTrajectorySeedProxyBuilder();

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWTrajectorySeedProxyBuilder(const FWTrajectorySeedProxyBuilder&); // stop default

   const FWTrajectorySeedProxyBuilder& operator=(const FWTrajectorySeedProxyBuilder&); // stop default

   using FWSimpleProxyBuilderTemplate<TrajectorySeed>::build;
   void build(const TrajectorySeed& iData, unsigned int iIndex,TEveElement& oItemHolder, const FWViewContext*) override;
};

FWTrajectorySeedProxyBuilder::FWTrajectorySeedProxyBuilder()
{
}

FWTrajectorySeedProxyBuilder::~FWTrajectorySeedProxyBuilder()
{
}

void
FWTrajectorySeedProxyBuilder::build( const TrajectorySeed& iData, unsigned int iIndex,TEveElement& itemHolder , const FWViewContext*) 
{
   // LocalPoint pnt = iData.startingState().parameters().position();
   // std::cout << pnt << std::endl;
   // std::cout << dynamic_cast<const SiPixelRecHit *>(&(*iData.recHits().first)) << std::endl;	
   // TEveVector startPos(pnt.x(), pnt.y(), pnt.z());

   TEvePointSet* pointSet = new TEvePointSet;
   TEveLine* line = new TEveLine;
   TEveStraightLineSet* lineSet = new TEveStraightLineSet;
   TrajectorySeed::const_iterator hit = iData.recHits().first;

   for(; hit != iData.recHits().second; hit++) {	 
	
      unsigned int id = hit->geographicalId();
      const FWGeometry *geom = item()->getGeom();
      const float* pars = geom->getParameters( id );
      const SiPixelRecHit * rh = dynamic_cast<const SiPixelRecHit *>(&*hit);
      // std::cout << id << "id "<< 	std::endl;
      if(rh){ 
         const SiPixelCluster * itc = rh->cluster().get();
         if( ! geom->contains( id ))
         {
            fwLog( fwlog::kWarning ) 
               << "failed get geometry of SiPixelCluster with detid: "
               << id << std::endl;
            continue;
         }


         float localPoint[3] = 
            {     
               fireworks::pixelLocalX(( *itc ).minPixelRow(), ( int )pars[0] ),
               fireworks::pixelLocalY(( *itc ).minPixelCol(), ( int )pars[1] ),
               0.0
            };

         float globalPoint[3];
         geom->localToGlobal( id, localPoint, globalPoint );

         pointSet->SetNextPoint( globalPoint[0], globalPoint[1], globalPoint[2] );
         line->SetNextPoint( globalPoint[0], globalPoint[1], globalPoint[2] );

      }

      else {
         const SiStripCluster *cluster = fireworks::extractClusterFromTrackingRecHit( &*hit );

         if (cluster) {
	 short firststrip = cluster->firstStrip();
         float localTop[3] = { 0.0, 0.0, 0.0 };
	 float localBottom[3] = { 0.0, 0.0, 0.0 };

         fireworks::localSiStrip( firststrip, localTop, localBottom, pars, id );

         float globalTop[3];
         float globalBottom[3];
         geom->localToGlobal( id, localTop, globalTop, localBottom, globalBottom );
  
         lineSet->AddLine( globalTop[0], globalTop[1], globalTop[2],
                           globalBottom[0], globalBottom[1], globalBottom[2] );
         }
      }
   }

   setupAddElement( pointSet, &itemHolder );
   setupAddElement( line, &itemHolder );
   setupAddElement( lineSet, &itemHolder );
}

//
// static member functions
//
REGISTER_FWPROXYBUILDER(FWTrajectorySeedProxyBuilder, TrajectorySeed, "TrajectorySeeds", FWViewType::kAll3DBits | FWViewType::kAllRPZBits);
