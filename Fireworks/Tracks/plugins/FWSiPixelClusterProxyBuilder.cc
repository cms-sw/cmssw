
// -*- C++ -*-
//
// Package:     Tracks
// Class  :     FWSiPixelClusterProxyBuilder
//
//
// Original Author:
//         Created:  Thu Dec  6 18:01:21 PST 2007
// $Id: FWSiPixelClusterProxyBuilder.cc,v 1.11 2010/05/05 10:42:18 mccauley Exp $
//

#include "TEvePointSet.h"

#include "Fireworks/Core/interface/fwLog.h"
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
  
  if ( ! pixels ) 
  {    
    fwLog(fwlog::kWarning)<<"ERROR: failed get SiPixelDigis"<<std::endl;
    return;
  }
  
  int row, column;

  double lx;
  double ly;

  for( SiPixelClusterCollectionNew::const_iterator set = pixels->begin(), setEnd = pixels->end();
       set != setEnd; ++set ) 
  {    
    unsigned int id = set->detId();
    DetId detid(id);
      
    const TGeoHMatrix* matrix = iItem->getGeom()->getMatrix( detid );
      
    if ( ! matrix ) 
    {
      fwLog(fwlog::kWarning) 
        <<"ERROR: failed get geometry of SiPixelCluster with detid: "
        << detid << std::endl;
      return;
    }

    const edmNew::DetSet<SiPixelCluster> & clusters = *set;
      
    for( edmNew::DetSet<SiPixelCluster>::const_iterator itc = clusters.begin(), edc = clusters.end(); 
         itc != edc; ++itc ) 
    {
      TEvePointSet* pointSet = new TEvePointSet();
      pointSet->SetMarkerSize(1);
      pointSet->SetMarkerStyle(4);
      pointSet->SetMarkerColor(iItem->defaultDisplayProperties().color());

      row = (*itc).minPixelRow();
      column = (*itc).minPixelCol();

      lx = 0.0;
      ly = 0.0;

      fireworks::pixelLocalXY(row, column, detid, lx, ly);
        
      double localPoint[3] = 
        {     
          lx, ly, 0.0
        };

      double globalPoint[3];
        
      matrix->LocalToMaster(localPoint, globalPoint);
      
      pointSet->SetNextPoint(globalPoint[0], globalPoint[1], globalPoint[2]);
      setupAddElement(pointSet, product); 
    }
  }    
}

REGISTER_FWPROXYBUILDER( FWSiPixelClusterProxyBuilder, SiPixelClusterCollectionNew, "SiPixelCluster", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
