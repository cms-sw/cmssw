#include "Fireworks/Core/interface/FWDigitSetProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

class FWCaloTowerEtProxyBuilder :  public FWDigitSetProxyBuilder
{
public:
   FWCaloTowerEtProxyBuilder( void ) {}  
   virtual ~FWCaloTowerEtProxyBuilder( void ) {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWCaloTowerEtProxyBuilder( const FWCaloTowerEtProxyBuilder& ); 			// stop default
   const FWCaloTowerEtProxyBuilder& operator=( const FWCaloTowerEtProxyBuilder& ); 	// stop default

   virtual void build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* );	
};

void
FWCaloTowerEtProxyBuilder::build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* )
{
   const CaloTowerCollection* collection = 0;
   iItem->get( collection );
   if( ! collection )
      return;

   TEveBoxSet* boxSet = addBoxSetToProduct(product);
   for (std::vector<CaloTower>::const_iterator it = collection->begin() ; it != collection->end(); ++it)
   {  
      const float* corners = item()->getGeom()->getCorners((*it).id().rawId());
      if (corners == 0) 
         continue;

      std::vector<float> scaledCorners(24);
      fireworks::energyTower3DCorners( corners, (*it).et(), scaledCorners );

      addBox( boxSet, &scaledCorners[0] );
   }
}

REGISTER_FWPROXYBUILDER( FWCaloTowerEtProxyBuilder, CaloTowerCollection, "Calo Tower Et", FWViewType::kISpyBit );

/*
void
FWCaloTowerEtProxyBuilder::build( const CaloTower& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* ) 
{
  const float* corners = item()->getGeom()->getCorners( iData.id().rawId() );
  if( corners == 0 ) {
    return;
  }
  
  fireworks::drawEnergyTower3D( corners, iData.et(), &oItemHolder, this, false );
}
*/
