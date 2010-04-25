// -*- C++ -*-
//
// Package:     Tracks
// Class  :     FWSiPixelClusterProxyBuilder
//
//
// Original Author:
//         Created:  Thu Dec  6 18:01:21 PST 2007
// $Id: FWSiPixelClusterProxyBuilder.cc,v 1.7 2010/04/23 21:02:00 amraktad Exp $
//

// system include files
#include "TEveManager.h"
#include "TEveCompound.h"
#include "TEveGeoNode.h"
#include "TEveStraightLineSet.h"

// user include files
#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"
#include "DataFormats/DetId/interface/DetId.h"

class FWSiPixelClusterProxyBuilder : public FWProxyBuilderBase
{
public:
   FWSiPixelClusterProxyBuilder() {}
   virtual ~FWSiPixelClusterProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();
private:
   virtual void build( const FWEventItem* iItem, TEveElementList* product );
   FWSiPixelClusterProxyBuilder( const FWSiPixelClusterProxyBuilder& );    // stop default
   const FWSiPixelClusterProxyBuilder& operator=( const FWSiPixelClusterProxyBuilder& );    // stop default
   //void modelChanges( const FWModelIds& iIds, TEveElement* iElements, int);
   //void applyChangesToAllModels( TEveElement* iElements, int );

protected:
   enum Mode { Clusters, Modules };
   virtual Mode getMode() { return Clusters; }
};

//______________________________________________________________________________

void FWSiPixelClusterProxyBuilder::build( const FWEventItem* iItem, TEveElementList* product )
{
   const SiPixelClusterCollectionNew* pixels = 0;
   iItem->get( pixels );
   if( 0 == pixels ) return;
   
   TEveCompound* top = createCompound();
   int index(0);
   for( SiPixelClusterCollectionNew::const_iterator set = pixels->begin(), setEnd = pixels->end();
       set != setEnd; ++set, ++index ) {
      TEveCompound* compound = createCompound();
      unsigned int id = set->detId();
      DetId detid(id);
      
      if( iItem->getGeom() ) {
         Mode m = getMode();
         if( m == Clusters ) {
            const TGeoHMatrix* matrix = iItem->getGeom()->getMatrix( detid );
            std::vector<TVector3> pixelPoints;
            const edmNew::DetSet<SiPixelCluster> & clusters = *set;
            for( edmNew::DetSet<SiPixelCluster>::const_iterator itc = clusters.begin(), edc = clusters.end(); itc != edc; ++itc ) {
               fireworks::pushPixelCluster( pixelPoints, matrix, detid, *itc );
            }
            fireworks::addTrackerHits3D( pixelPoints, compound, iItem->defaultDisplayProperties().color(), 1 );
         } else if( m == Modules ) {
            TEveGeoShape* shape = iItem->getGeom()->getShape( id );
            if( 0 != shape ) {
               shape->SetMainTransparency( 50 );
               setupAddElement( shape, compound );
            }
         }
      }
      setupAddElement(compound, top);
   }
   
   setupAddElement(top, product);
}

class FWSiPixelClusterModProxyBuilder : public FWSiPixelClusterProxyBuilder {
public:
   FWSiPixelClusterModProxyBuilder() {}
   virtual ~FWSiPixelClusterModProxyBuilder() {}
   
   REGISTER_PROXYBUILDER_METHODS();
protected:
   virtual Mode getMode() { return Modules; }
};

REGISTER_FWPROXYBUILDER( FWSiPixelClusterProxyBuilder, SiPixelClusterCollectionNew, "SiPixelCluster", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
REGISTER_FWPROXYBUILDER( FWSiPixelClusterModProxyBuilder, SiPixelClusterCollectionNew, "SiPixelClusterDets", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
