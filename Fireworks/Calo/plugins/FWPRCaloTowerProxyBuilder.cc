#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"

class FWPRCaloTowerProxyBuilder : public FWSimpleProxyBuilderTemplate<CaloTower>
{
public:
   FWPRCaloTowerProxyBuilder( void ) {}  
   virtual ~FWPRCaloTowerProxyBuilder( void ) {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWPRCaloTowerProxyBuilder( const FWPRCaloTowerProxyBuilder& ); 			// stop default
   const FWPRCaloTowerProxyBuilder& operator=( const FWPRCaloTowerProxyBuilder& ); 	// stop default

   void build( const CaloTower& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* );
};

void
FWPRCaloTowerProxyBuilder::build( const CaloTower& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* ) 
{
  const float* corners = item()->getGeom()->getCorners( iData.id().rawId() );
  if( corners == 0 ) {
    return;
  }
  
  fireworks::drawEnergyTower3D( corners, iData.et(), &oItemHolder, this, false );
}

REGISTER_FWPROXYBUILDER( FWPRCaloTowerProxyBuilder, CaloTower, "CaloTower", FWViewType::kISpyBit );
