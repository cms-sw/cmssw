// -*- C++ -*-
//
// Package:     Tracks
// Class  :     FWSiPixelClusterProxyBuilder
//
//
// Original Author:
//         Created:  Thu Dec  6 18:01:21 PST 2007
// $Id: FWSiPixelClusterProxyBuilder.cc,v 1.9 2010/05/03 15:47:44 amraktad Exp $
//

#include "TEveCompound.h"

#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/DetId/interface/DetId.h"

class FWSiPixelClusterProxyBuilder : public FWProxyBuilderBase
{
public:
   FWSiPixelClusterProxyBuilder() {}
   virtual ~FWSiPixelClusterProxyBuilder() {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   virtual void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*);
   FWSiPixelClusterProxyBuilder( const FWSiPixelClusterProxyBuilder& );
   const FWSiPixelClusterProxyBuilder& operator=( const FWSiPixelClusterProxyBuilder& );
};

//______________________________________________________________________________

void FWSiPixelClusterProxyBuilder::build( const FWEventItem* iItem, TEveElementList* product , const FWViewContext*)
{
  const SiPixelClusterCollectionNew* pixels = 0;
  
  iItem->get(pixels);
  
  if( 0 == pixels ) 
    return;

  for( SiPixelClusterCollectionNew::const_iterator set = pixels->begin(), setEnd = pixels->end();
       set != setEnd; ++set ) 
  {   
    TEveCompound* compound = createCompound();
 
    unsigned int id = set->detId();
    DetId detid(id);
      
    if( iItem->getGeom() ) 
    {
      const TGeoHMatrix* matrix = iItem->getGeom()->getMatrix( detid );
      std::vector<TVector3> pixelPoints;
      
      const edmNew::DetSet<SiPixelCluster> & clusters = *set;
       
      for( edmNew::DetSet<SiPixelCluster>::const_iterator itc = clusters.begin(), edc = clusters.end(); 
           itc != edc; ++itc ) 
        fireworks::pushPixelCluster( pixelPoints, matrix, detid, *itc );
      
      
      fireworks::addTrackerHits3D( pixelPoints, compound, iItem->defaultDisplayProperties().color(), 1 );
    }
  
    setupAddElement(compound, product);
  }    
}

REGISTER_FWPROXYBUILDER( FWSiPixelClusterProxyBuilder, SiPixelClusterCollectionNew, "SiPixelCluster", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
