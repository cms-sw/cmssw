// -*- C++ -*-
// $Id: FWTracksRecHitsProxyBuilder.cc,v 1.1 2009/01/16 10:37:00 Tom Danielson
//

// user include files
#include "TEveGeoShape.h"
#include "TEvePointSet.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"


#include "Fireworks/Core/interface/fwLog.h"

class FWTracksRecHitsProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Track>
{
public:
   FWTracksRecHitsProxyBuilder() {}
   virtual ~FWTracksRecHitsProxyBuilder() {}
  
   REGISTER_PROXYBUILDER_METHODS();
  
   static bool representsSubPart();
   
private:
   void build(const reco::Track& iData, unsigned int iIndex,TEveElement& oItemHolder, const FWViewContext*);

   FWTracksRecHitsProxyBuilder(const FWTracksRecHitsProxyBuilder&);    // stop default
   const FWTracksRecHitsProxyBuilder& operator=(const FWTracksRecHitsProxyBuilder&);    // stop default
};


void
FWTracksRecHitsProxyBuilder::build(const reco::Track& track, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*) 
{
   if( track.extra().isAvailable() )
   {
      std::vector<TVector3> points;
      const FWEventItem &iItem = *item();
      fireworks::pushPixelHits(points, iItem, track);
    
      TEvePointSet* pointSet = new TEvePointSet();
      for( std::vector<TVector3>::const_iterator it = points.begin(), itEnd = points.end(); it != itEnd; ++it) {
         pointSet->SetNextPoint(it->x(), it->y(), it->z());
      }
      setupAddElement(pointSet, &oItemHolder);
      fireworks::addSiStripClusters(item(), track, &oItemHolder, false, true);
   }
}

bool FWTracksRecHitsProxyBuilder::representsSubPart()
{
   return true;
}

REGISTER_FWPROXYBUILDER(FWTracksRecHitsProxyBuilder, reco::Track, "TrackHits", FWViewType::kAll3DBits | FWViewType::kAllRPZBits);


//______________________________________________________________________________


class FWTracksModulesProxyBuilder : public FWSimpleProxyBuilderTemplate<reco::Track>
{
public:
   FWTracksModulesProxyBuilder() {}
   virtual ~FWTracksModulesProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();
  
   static bool representsSubPart();
private:
   void build(const reco::Track& iData, unsigned int iIndex,TEveElement& oItemHolder, const FWViewContext*);

   FWTracksModulesProxyBuilder(const FWTracksModulesProxyBuilder&);    // stop default
   const FWTracksModulesProxyBuilder& operator=(const FWTracksModulesProxyBuilder&);    // stop default
};

void
FWTracksModulesProxyBuilder::build(const reco::Track& track, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*) 
{
   if( track.extra().isAvailable() )
   {
      for( trackingRecHit_iterator recIt = track.recHitsBegin(), recItEnd = track.recHitsEnd();
           recIt != recItEnd; ++recIt )
      {
         DetId detid = (*recIt)->geographicalId();
         if ((*recIt)->isValid())
         {
            if (detid.det() ==  DetId::Muon)
            {
               if (detid.subdetId() == MuonSubdetId::DT)
                  detid = DetId(DTChamberId(detid)); // get rid of layer bits
            }

            TEveGeoShape* shape = item()->getGeom()->getShape( detid );
            if (shape)
            {
               setupAddElement(shape, &oItemHolder);
            }
            else
            {
               fwLog(fwlog::kDebug) <<  "Failed to get shape extract for track-id "<< iIndex <<", tracking rec hit: "
                                    << "\n" << fireworks::info(detid) << std::endl;
            }
         }
      }
   }
}

bool FWTracksModulesProxyBuilder::representsSubPart()
{
   return true;
}

REGISTER_FWPROXYBUILDER(FWTracksModulesProxyBuilder, reco::Track, "TrackDets", FWViewType::kAll3DBits | FWViewType::kAllRPZBits);
