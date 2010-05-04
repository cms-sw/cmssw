// -*- C++ -*-
//
// Package:     Tracks
// Class  :     FWSiPixelClusterDetProxyBuilder
//
//
// Original Author:
//         Created:  Thu Dec  6 18:01:21 PST 2007
// $Id: FWSiPixelClusterDetProxyBuilder.cc,v 1.9 2010/05/03 15:47:44 amraktad Exp $
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
  
  if( 0 == pixels ) 
    return;
    
  for( SiPixelClusterCollectionNew::const_iterator set = pixels->begin(), setEnd = pixels->end();
       set != setEnd; ++set) 
  {
    TEveCompound* compound = createCompound();
    unsigned int id = set->detId();
    DetId detid(id);
      
    if( iItem->getGeom() ) 
    {
      TEveGeoShape* shape = iItem->getGeom()->getShape( id );
        
      if( 0 != shape ) 
      {
        shape->SetMainTransparency( 50 );
        setupAddElement( shape, compound );
      }
    }
    
    setupAddElement(compound, product);
  }
}

REGISTER_FWPROXYBUILDER( FWSiPixelClusterDetProxyBuilder, SiPixelClusterCollectionNew, "SiPixelClusterDets", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
