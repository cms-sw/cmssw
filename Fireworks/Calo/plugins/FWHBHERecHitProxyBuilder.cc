#include "TEveCompound.h"
#include "TEveBoxSet.h"

#include "Fireworks/Core/interface/FWDigitSetProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

class FWHBHERecHitProxyBuilder : public FWDigitSetProxyBuilder
{
public:
   FWHBHERecHitProxyBuilder( void )
    {}
  
   virtual ~FWHBHERecHitProxyBuilder( void ) 
    {}

   virtual bool havePerViewProduct(FWViewType::EType) const { return true; }
   virtual void scaleProduct(TEveElementList* parent, FWViewType::EType, const FWViewContext* vc);

   REGISTER_PROXYBUILDER_METHODS();

private:
   virtual void build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* );

   FWHBHERecHitProxyBuilder( const FWHBHERecHitProxyBuilder& );
   const FWHBHERecHitProxyBuilder& operator=( const FWHBHERecHitProxyBuilder& );
};

namespace
{

float  scaleFactor(const FWViewContext* vc)
{
   // printf("scale face %f \n", vc->getEnergyScale()->getScaleFactor3D());
   return vc->getEnergyScale()->getScaleFactor3D()/20;
}

void viewContextBoxScale( const float* corners, float scale, bool plotEt, std::vector<float>& scaledCorners, bool invert)
{
   TEveVector  center;
   for( unsigned int i = 0; i < 24; i += 3 )
   {	 
      center[0] += corners[i];
      center[1] += corners[i + 1];
      center[2] += corners[i + 2];
   }
   center *= 1.f/8.f;

   if (plotEt)
   {
      scale *= center.Perp()/center.Mag();
   }

   // Coordinates for a scaled version of the original box
   for( unsigned int i = 0; i < 24; i += 3 )
   {	
      scaledCorners[i] = center[0] + ( corners[i] - center[0] ) * scale;
      scaledCorners[i + 1] = center[1] + ( corners[i + 1] - center[1] ) * scale;
      scaledCorners[i + 2] = center[2] + ( corners[i + 2] - center[2] ) * scale;
   }
      
   if( invert )
      fireworks::invertBox( scaledCorners );
}
}

void
FWHBHERecHitProxyBuilder::scaleProduct(TEveElementList* parent, FWViewType::EType type, const FWViewContext* vc)
{

   const HBHERecHitCollection* collection = 0;
   item()->get( collection );
   if (! collection)
      return;

   int index = 0;
   std::vector<float> scaledCorners(24);

   float scale = scaleFactor(vc);

   for (std::vector<HBHERecHit>::const_iterator it = collection->begin() ; it != collection->end(); ++it, ++index)
   {
      const float* corners = item()->getGeom()->getCorners((*it).detid());
      if (corners == 0) 
         continue;
      FWDigitSetProxyBuilder::BFreeBox_t* b = (FWDigitSetProxyBuilder::BFreeBox_t*)getBoxSet()->GetPlex()->Atom(index);

      /*          
      printf("--------------------scale product \n");
      for (int i = 0; i < 8 ; ++i)
         printf("[%f %f %f ]\n",b->fVertices[i][0], b->fVertices[i][1],b->fVertices[i][2] );
      
      */

      viewContextBoxScale(corners, (*it).energy()*scale, vc->getEnergyScale()->getPlotEt(), scaledCorners, true);
      /*
        printf("after \n");
        for (int i = 0; i < 8 ; ++i)
        printf("[%f %f %f ]\n",b->fVertices[i][0], b->fVertices[i][1],b->fVertices[i][2] );        
      */
      memcpy(b->fVertices, &scaledCorners[0], sizeof(b->fVertices));
   }
   getBoxSet()->ElementChanged();
}

//______________________________________________________________________________

void
FWHBHERecHitProxyBuilder::build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* vc)
{
   const HBHERecHitCollection* collection = 0;
   iItem->get( collection );

   if( 0 == collection )
      return;

   std::vector<HBHERecHit>::const_iterator it = collection->begin();
   std::vector<HBHERecHit>::const_iterator itEnd = collection->end();
   std::vector<float> scaledCorners(24);

   TEveBoxSet* boxSet = addBoxSetToProduct(product);
   int index = 0;

   float scale = scaleFactor(vc);

   TEveVector centre;
   for (std::vector<HBHERecHit>::const_iterator it = collection->begin() ; it != collection->end(); ++it)
   {  
      const float* corners = context().getGeom()->getCorners((*it).detid());
      if (corners)
      {
         for( unsigned int i = 0; i < 24; i += 3 )
         {	 
            centre[0] += corners[i];
            centre[1] += corners[i + 1];
            centre[2] += corners[i + 2];
         }
         centre.Normalize();
         context().voteMaxEtAndEnergy( centre.Perp() *it->energy() ,it->energy());
         viewContextBoxScale(corners, (*it).energy()*scale, vc->getEnergyScale()->getPlotEt(), scaledCorners, true);
      }

      addBox(boxSet, &scaledCorners[0], iItem->modelInfo(index++).displayProperties());
   }
}

REGISTER_FWPROXYBUILDER( FWHBHERecHitProxyBuilder, HBHERecHitCollection, "HBHE RecHit", FWViewType::kISpyBit );
