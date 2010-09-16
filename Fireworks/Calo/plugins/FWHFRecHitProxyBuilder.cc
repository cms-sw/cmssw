#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "TEveCompound.h"

class FWHFRecHitProxyBuilder : public FWProxyBuilderBase
{
public:
   FWHFRecHitProxyBuilder( void ) 
     : m_maxEnergy( 5.0 )
    {}
  
   virtual ~FWHFRecHitProxyBuilder( void ) 
    {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   virtual void build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* );

   Float_t m_maxEnergy;

   // Disable default copy constructor
   FWHFRecHitProxyBuilder( const FWHFRecHitProxyBuilder& );
   // Disable default assignment operator
   const FWHFRecHitProxyBuilder& operator=( const FWHFRecHitProxyBuilder& );
};

void
FWHFRecHitProxyBuilder::build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* )
{
   const HFRecHitCollection* collection = 0;
   iItem->get( collection );

   if( 0 == collection )
   {
      return;
   }

   const FWGeometry *geom = iItem->getGeom();

   std::vector<HFRecHit>::const_iterator it = collection->begin();
   std::vector<HFRecHit>::const_iterator itEnd = collection->end();
   for( ; it != itEnd; ++it )
   {
      if(( *it ).energy() > m_maxEnergy )
	m_maxEnergy = ( *it ).energy();
   }
   
   for( it = collection->begin(); it != itEnd; ++it )
   {
      const float* corners = geom->getCorners(( *it ).detid().rawId());
      if( corners == 0 )
      {
	 TEveCompound* compound = createCompound();
	 setupAddElement( compound, product );

	 continue;
      }
      
      fireworks::drawEnergyScaledBox3D( corners, ( *it ).energy() / m_maxEnergy, product, this, true );
   }
}

REGISTER_FWPROXYBUILDER( FWHFRecHitProxyBuilder, HFRecHitCollection, "HF RecHit", FWViewType::kISpyBit );
