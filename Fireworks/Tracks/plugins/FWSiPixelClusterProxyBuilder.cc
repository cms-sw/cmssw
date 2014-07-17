
// -*- C++ -*-
//
// Package:     Tracks
// Class  :     FWSiPixelClusterProxyBuilder
//
//
// Original Author:
//         Created:  Thu Dec  6 18:01:21 PST 2007
//

#include "TEvePointSet.h"
#include "TEveStraightLineSet.h"
#include "TEveCompound.h"
#include "TEveBox.h"
#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

class FWSiPixelClusterProxyBuilder : public FWProxyBuilderBase
{
public:
  FWSiPixelClusterProxyBuilder( void ) {}
  virtual ~FWSiPixelClusterProxyBuilder( void ) {}

  REGISTER_PROXYBUILDER_METHODS();

private:
  // Disable default copy constructor
  FWSiPixelClusterProxyBuilder( const FWSiPixelClusterProxyBuilder& );
  // Disable default assignment operator
  const FWSiPixelClusterProxyBuilder& operator=( const FWSiPixelClusterProxyBuilder& );

  using FWProxyBuilderBase::build;
  virtual void build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* ) override;
};

void
FWSiPixelClusterProxyBuilder::build( const FWEventItem* iItem, TEveElementList* product , const FWViewContext* )
{
   const SiPixelClusterCollectionNew* pixels = 0;
  
   iItem->get( pixels );
  
   if( ! pixels ) 
   {    
      fwLog( fwlog::kWarning ) << "failed get SiPixelDigis" << std::endl;
      return;
   }

   for( SiPixelClusterCollectionNew::const_iterator set = pixels->begin(), setEnd = pixels->end();
        set != setEnd; ++set ) 
   {    
      unsigned int id = set->detId();

      const FWGeometry *geom = iItem->getGeom();
      const float* pars = geom->getParameters( id );

      const edmNew::DetSet<SiPixelCluster> & clusters = *set;
      
      for( edmNew::DetSet<SiPixelCluster>::const_iterator itc = clusters.begin(), edc = clusters.end(); 
           itc != edc; ++itc ) 
      {
         TEveElement* itemHolder = createCompound();
         product->AddElement(itemHolder);

         TEvePointSet* pointSet = new TEvePointSet;
      
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

         setupAddElement( pointSet, itemHolder );

         TEveStraightLineSet* ls = new TEveStraightLineSet();
         for(int j=0;j< (*itc).size(); j++ )
         {

            //            float adc= (*itc).pixel(j).adc*0.03/5000.;
            float adc= 0.025;
            float offsetx[4] = {-0.4,-0.4,+0.4,+0.4};
            float offsety[4] = {-0.4,+0.4,+0.4,-0.4};
            //            float vert[24];

            std::vector<TEveVector> boxCorners;
            for(int of=0;of<8;of++) {
               float lp[3]= {
                  fireworks::pixelLocalX(
                                         (*itc).pixel(j).x+offsetx[of%4],( int )pars[0]) ,
                  fireworks::pixelLocalY(
                                         (*itc).pixel(j).y+offsety[of%4], ( int )pars[1] ),
                  (of<4)?(0.0f):(adc)
               };

               TEveVector p;
               geom->localToGlobal( id,lp,  p.Arr() );

               boxCorners.push_back(p);

            }

            // bottom
            ls->AddLine(boxCorners[0], boxCorners[1]);
            ls->AddLine(boxCorners[1], boxCorners[2]);
            ls->AddLine(boxCorners[2], boxCorners[3]);
            ls->AddLine(boxCorners[3], boxCorners[0]);
            // top
            ls->AddLine(boxCorners[4], boxCorners[5]);
            ls->AddLine(boxCorners[5], boxCorners[6]);
            ls->AddLine(boxCorners[6], boxCorners[7]);
            ls->AddLine(boxCorners[7], boxCorners[4]);
            // sides
            ls->AddLine(boxCorners[0], boxCorners[4]);
            ls->AddLine(boxCorners[1], boxCorners[5]);
            ls->AddLine(boxCorners[2], boxCorners[6]);
            ls->AddLine(boxCorners[3], boxCorners[7]);


         }

         setupAddElement( ls, itemHolder );

      }
   }    
}

REGISTER_FWPROXYBUILDER( FWSiPixelClusterProxyBuilder, SiPixelClusterCollectionNew, "SiPixelCluster", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
