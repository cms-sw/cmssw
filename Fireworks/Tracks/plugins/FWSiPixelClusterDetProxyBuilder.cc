// -*- C++ -*-
//
// Package:     Tracks
// Class  :     FWSiPixelClusterDetProxyBuilder
//
//
// Original Author:
//         Created:  Thu Dec  6 18:01:21 PST 2007
// $Id: FWSiPixelClusterDetProxyBuilder.cc,v 1.1 2010/05/04 17:52:34 mccauley Exp $
//

#include "TEveCompound.h"
#include "TEveGeoNode.h"

#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/DetId/interface/DetId.h"

class FWSiPixelClusterDetProxyBuilder : public FWProxyBuilderBase
{
public:
  FWSiPixelClusterDetProxyBuilder() {}
  virtual ~FWSiPixelClusterDetProxyBuilder() {}
  
  REGISTER_PROXYBUILDER_METHODS();

private:
  virtual void build(const FWEventItem* iItem, TEveElementList* product, const FWViewContext*);
  FWSiPixelClusterDetProxyBuilder(const FWSiPixelClusterDetProxyBuilder&);
  const FWSiPixelClusterDetProxyBuilder& operator=(const FWSiPixelClusterDetProxyBuilder&);
};

void FWSiPixelClusterDetProxyBuilder::build( const FWEventItem* iItem, TEveElementList* product , const FWViewContext*)
{
  const SiPixelClusterCollectionNew* pixels = 0;
  
  iItem->get(pixels);
  
  if( ! pixels ) 
    return;
    
  for( SiPixelClusterCollectionNew::const_iterator set = pixels->begin(), setEnd = pixels->end();
       set != setEnd; ++set) 
  {
    unsigned int id = set->detId();
    DetId detid(id);
      
    if( iItem->getGeom() ) 
    {
      const edmNew::DetSet<SiPixelCluster> & clusters = *set;
      
      for( edmNew::DetSet<SiPixelCluster>::const_iterator itc = clusters.begin(), edc = clusters.end(); 
           itc != edc; ++itc ) 
      {
        TEveGeoShape* shape = iItem->getGeom()->getShape(detid);
       
        if ( shape )
        {
          shape->SetMainTransparency(50);
          setupAddElement(shape, product);
        }
      }
    }
  }
}

REGISTER_FWPROXYBUILDER( FWSiPixelClusterDetProxyBuilder, SiPixelClusterCollectionNew, "SiPixelClusterDets", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
