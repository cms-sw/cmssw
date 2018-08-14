// -*- C++ -*-
// $Id: FWTracksModulesProxyBuilder.cc,v 1.1 2009/01/16 10:37:00 Tom Danielson
//

// user include files
#include "TEveGeoShape.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"

#include "Fireworks/Core/interface/fwLog.h"

class FWTracksModulesProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Track>
{
public:
   FWTracksModulesProxyBuilder( void ) {}
   ~FWTracksModulesProxyBuilder( void ) override {}

   REGISTER_PROXYBUILDER_METHODS();
  
   static bool representsSubPart( void );
private:
   using FWSimpleProxyBuilderTemplate<reco::Track>::build;
   void build( const reco::Track& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* ) override;

   FWTracksModulesProxyBuilder( const FWTracksModulesProxyBuilder& ) = delete;    // stop default
   const FWTracksModulesProxyBuilder& operator=( const FWTracksModulesProxyBuilder& ) = delete;    // stop default
};

void
FWTracksModulesProxyBuilder::build( const reco::Track& track, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* ) 
{
   if( track.extra().isAvailable() )
   {
     const FWGeometry *geom = item()->getGeom();

      for( trackingRecHit_iterator recIt = track.recHitsBegin(), recItEnd = track.recHitsEnd();
           recIt != recItEnd; ++recIt )
      {
         DetId detid = ( *recIt )->geographicalId();
         if(( *recIt )->isValid())
         {
            if( detid.det() ==  DetId::Muon )
            {
               if( detid.subdetId() == MuonSubdetId::DT )
		  detid = DetId( DTChamberId( detid )); // get rid of layer bits
            }

            TEveGeoShape* shape = geom->getEveShape( detid );
            if( shape )
            {
               setupAddElement( shape, &oItemHolder );
            }
            else
            {
               fwLog( fwlog::kDebug )
		 << "Failed to get shape extract for track-id " << iIndex << ", tracking rec hit: "
		 << "\n" << fireworks::info( detid ) << std::endl;
            }
         }
      }
   }
}

bool
FWTracksModulesProxyBuilder::representsSubPart( void )
{
   return true;
}

REGISTER_FWPROXYBUILDER( FWTracksModulesProxyBuilder, reco::Track, "TrackDets", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
