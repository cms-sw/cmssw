#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "TEveCompound.h"

class FWHBHERecHitEtProxyBuilder : public FWProxyBuilderBase
{
public:
   FWHBHERecHitEtProxyBuilder( void )
     : m_maxEnergy( 0.85 )
    {}
  
   virtual ~FWHBHERecHitEtProxyBuilder( void ) 
    {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   virtual void build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* );

   Float_t m_maxEnergy;

   // Disable default copy constructor
   FWHBHERecHitEtProxyBuilder( const FWHBHERecHitEtProxyBuilder& );
   // Disable default assignment operator
   const FWHBHERecHitEtProxyBuilder& operator=( const FWHBHERecHitEtProxyBuilder& );
};

void
FWHBHERecHitEtProxyBuilder::build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* )
{
   const HBHERecHitCollection* collection = 0;
   iItem->get( collection );

   if( 0 == collection )
   {
      return;
   }

   const FWGeometry *geom = iItem->getGeom();

   std::vector<HBHERecHit>::const_iterator it = collection->begin();
   std::vector<HBHERecHit>::const_iterator itEnd = collection->end();
   for( ; it != itEnd; ++it )
   {
      if(( *it ).energy() > m_maxEnergy )
         m_maxEnergy = ( *it ).energy();
   }
   
   for( it = collection->begin(); it != itEnd; ++it )
   {
      const float* corners = geom->getCorners(( *it ).detid());
      if( corners == 0 )
      {
	 TEveCompound* compound = createCompound();
	 setupAddElement( compound, product );

	 continue;
      }

      fireworks::drawEtScaledBox3D( corners, ( *it ).energy(), m_maxEnergy, product, this, true );
   }
}

REGISTER_FWPROXYBUILDER( FWHBHERecHitEtProxyBuilder, HBHERecHitCollection, "HBHE RecHit Et", FWViewType::kISpyBit );
