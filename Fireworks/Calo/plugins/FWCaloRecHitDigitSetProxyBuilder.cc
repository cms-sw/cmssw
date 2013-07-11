#include "Fireworks/Calo/plugins/FWCaloRecHitDigitSetProxyBuilder.h"
#include "TEveBoxSet.h"
#include "FWCaloRecHitDigitSetProxyBuilder.h"
#include "Fireworks/Core/interface/FWDigitSetProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"

#include "Fireworks/Core/interface/FWProxyBuilderConfiguration.h"

FWCaloRecHitDigitSetProxyBuilder::FWCaloRecHitDigitSetProxyBuilder()
   : m_invertBox(false), m_ignoreGeoShapeSize(false) 
{} 

//______________________________________________________________________________

void FWCaloRecHitDigitSetProxyBuilder::setItem(const FWEventItem* iItem)
{
   FWProxyBuilderBase::setItem(iItem);
   // if (iItem) iItem->getConfig()->assertParam( "IgnoreShapeSize", false);
}
//______________________________________________________________________________

void FWCaloRecHitDigitSetProxyBuilder::viewContextBoxScale( const float* corners, float scale, bool plotEt, std::vector<float>& scaledCorners, const CaloRecHit*)
{
   if ( m_ignoreGeoShapeSize)
   {
      // Same functionality as fireworks::energyTower3DCorners()

      for( int i = 0; i < 24; ++i )
         scaledCorners[i] = corners[i];

      // Coordinates of a front face scaled 
      if( m_invertBox )
      {
         // We know, that an ES rechit geometry in -Z needs correction. 
         // The back face is actually its front face.
         for( unsigned int i = 0; i < 12; i += 3 )
         {
            m_vector.Set( corners[i] - corners[i + 12], corners[i + 1] - corners[i + 13], corners[i + 2] - corners[i + 14] );
            m_vector.Normalize();
            m_vector *= scale;
	    
            scaledCorners[i] = corners[i] + m_vector.fX;
            scaledCorners[i + 1] = corners[i + 1] + m_vector.fY;
            scaledCorners[i + 2] = corners[i + 2] + m_vector.fZ;
         }
      } 
      else
      {
         for( unsigned int i = 0; i < 12; i += 3 )
         {
            m_vector.Set( corners[i + 12] - corners[i], corners[i + 13] - corners[i + 1], corners[i + 14] - corners[i + 2] );
            m_vector.Normalize();
            m_vector *= scale;
	    
            scaledCorners[i] = corners[i + 12];
            scaledCorners[i + 1] = corners[i + 13];
            scaledCorners[i + 2] = corners[i + 14];
	    
            scaledCorners[i + 12] = corners[i + 12] + m_vector.fX;
            scaledCorners[i + 13] = corners[i + 13] + m_vector.fY;
            scaledCorners[i + 14] = corners[i + 14] + m_vector.fZ;
         }
      }
   }
   else {

      // Same functionality as fireworks::energyScaledBox3DCorners().

      m_vector.Set(0.f, 0.f, 0.f);
      for( unsigned int i = 0; i < 24; i += 3 )
      {	 
         m_vector[0] += corners[i];
         m_vector[1] += corners[i + 1];
         m_vector[2] += corners[i + 2];
      }
      m_vector *= 1.f/8.f;

      if (plotEt)
      {
         scale *= m_vector.Perp()/m_vector.Mag();
      }

      // Coordinates for a scaled version of the original box
      for( unsigned int i = 0; i < 24; i += 3 )
      {	
         scaledCorners[i] = m_vector[0] + ( corners[i] - m_vector[0] ) * scale;
         scaledCorners[i + 1] = m_vector[1] + ( corners[i + 1] - m_vector[1] ) * scale;
         scaledCorners[i + 2] = m_vector[2] + ( corners[i + 2] - m_vector[2] ) * scale;
      }
      
      if( m_invertBox )
         fireworks::invertBox( scaledCorners );
   }
}
//_____________________________________________________________________________

float  FWCaloRecHitDigitSetProxyBuilder::scaleFactor(const FWViewContext* vc)
{
   // printf("scale face %f \n", vc->getEnergyScale()->getScaleFactor3D());
     return vc->getEnergyScale()->getScaleFactor3D()/50;
}

//______________________________________________________________________________

void
FWCaloRecHitDigitSetProxyBuilder::scaleProduct(TEveElementList* parent, FWViewType::EType type, const FWViewContext* vc)
{
   size_t size = item()->size();
   if (!size) return;


   std::vector<float> scaledCorners(24);
   float scale = scaleFactor(vc);

   assert(parent->NumChildren() == 1);
   TEveBoxSet* boxSet = static_cast<TEveBoxSet*>(*parent->BeginChildren());

   for (int index = 0; index < static_cast<int>(size); ++index)
   {
      const CaloRecHit* hit = (const CaloRecHit*)item()->modelData(index);
      const float* corners = item()->getGeom()->getCorners(hit->detid());
      if (corners == 0)  continue;

      FWDigitSetProxyBuilder::BFreeBox_t* b = (FWDigitSetProxyBuilder::BFreeBox_t*)boxSet->GetPlex()->Atom(index);

      viewContextBoxScale(corners, hit->energy()*scale, vc->getEnergyScale()->getPlotEt(), scaledCorners, hit);
      memcpy(b->fVertices, &scaledCorners[0], sizeof(b->fVertices));
   }
   boxSet->ElementChanged();
}
//______________________________________________________________________________

void
FWCaloRecHitDigitSetProxyBuilder::build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* vc)
{
   size_t size = iItem->size();
   if (!size) return;

   // m_ignoreGeoShapeSize = item()->getConfig()->value<bool>("IgnoreShapeSize");

   std::vector<float> scaledCorners(24);

   float scale = scaleFactor(vc);

   TEveBoxSet* boxSet = addBoxSetToProduct(product);
   boxSet->SetAntiFlick(kTRUE);
   for (int index = 0; index < static_cast<int>(size); ++index)
   {  
      const CaloRecHit* hit = (const CaloRecHit*)item()->modelData(index);

      const float* corners = context().getGeom()->getCorners(hit->detid());
      if (corners)
      {
         m_vector.Set(0.f, 0.f, 0.f);
         for( unsigned int i = 0; i < 24; i += 3 )
         {	 
            m_vector[0] += corners[i];
            m_vector[1] += corners[i + 1];
            m_vector[2] += corners[i + 2];
         }
         m_vector.Normalize();
         context().voteMaxEtAndEnergy( m_vector.Perp() *hit->energy(), hit->energy());
         viewContextBoxScale( corners, hit->energy()*scale, vc->getEnergyScale()->getPlotEt(), scaledCorners, hit);
      }

      addBox(boxSet, &scaledCorners[0], iItem->modelInfo(index).displayProperties());
   }
}


