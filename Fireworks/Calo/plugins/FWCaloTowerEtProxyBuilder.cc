#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"

class FWCaloTowerEtProxyBuilder : public FWSimpleProxyBuilderTemplate<CaloTower>
{
public:
   FWCaloTowerEtProxyBuilder( void ) {}  
   virtual ~FWCaloTowerEtProxyBuilder( void ) {}

   REGISTER_PROXYBUILDER_METHODS();

private:
   FWCaloTowerEtProxyBuilder( const FWCaloTowerEtProxyBuilder& ); 			// stop default
   const FWCaloTowerEtProxyBuilder& operator=( const FWCaloTowerEtProxyBuilder& ); 	// stop default

   void build( const CaloTower& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* );
};

void
FWCaloTowerEtProxyBuilder::build( const CaloTower& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* ) 
{
  const float* corners = item()->getGeom()->getCorners( iData.id().rawId() );
  if( corners == 0 ) {
    return;
  }
  
  fireworks::drawEnergyTower3D( corners, iData.et(), &oItemHolder, this, false );
}

REGISTER_FWPROXYBUILDER( FWCaloTowerEtProxyBuilder, CaloTower, "Calo Tower", FWViewType::kISpyBit );
