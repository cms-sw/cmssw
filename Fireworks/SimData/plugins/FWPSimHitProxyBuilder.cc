/*
 *  FWPSimHitProxyBuilder.cc
 *  FWorks
 *
 *  Created by Ianna Osborne on 9/9/10.
 *
 */

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include <DataFormats/MuonDetId/interface/DTWireId.h>

#include "TEvePointSet.h"

class FWPSimHitProxyBuilder : public FWSimpleProxyBuilderTemplate<PSimHit>
{
public:
   FWPSimHitProxyBuilder( void ) {} 
   virtual ~FWPSimHitProxyBuilder( void ) {}

   virtual bool haveSingleProduct() const override { return false; }

   REGISTER_PROXYBUILDER_METHODS();

private:
   // Disable default copy constructor
   FWPSimHitProxyBuilder( const FWPSimHitProxyBuilder& );
   // Disable default assignment operator
   const FWPSimHitProxyBuilder& operator=( const FWPSimHitProxyBuilder& );

   void buildViewType( const PSimHit& iData, unsigned int iIndex, TEveElement& oItemHolder, FWViewType::EType type, const FWViewContext* ) override;
};

void
FWPSimHitProxyBuilder::buildViewType( const PSimHit& iData, unsigned int iIndex, TEveElement& oItemHolder, FWViewType::EType type, const FWViewContext* )
{
   TEvePointSet* pointSet = new TEvePointSet;
   setupAddElement( pointSet, &oItemHolder );
   const FWGeometry *geom = item()->getGeom();
   unsigned int rawid = iData.detUnitId();
   if( ! geom->contains( rawid ))
   {
      fwLog( fwlog::kError )
	<< "failed to get geometry of detid: " 
	<< rawid << std::endl;
      return;
   }
   
   float local[3] = { iData.localPosition().x(), iData.localPosition().y(), iData.localPosition().z() };
   float global[3];

   // Specialized for DT simhits
   DetId id(rawid);
   if (id.det()==DetId::Muon && id.subdetId()==1) {   
     DTWireId wId(rawid);
     rawid = wId.layerId().rawId(); // DT simhits are in the RF of the DTLayer, but their ID is the id of the wire!
     if (abs(iData.particleType())!=13){
       pointSet->SetMarkerStyle(26); // Draw non-muon simhits (e.g. delta rays) with a different marker
     } 
     if (type == FWViewType::kRhoZ) { // 
       // In RhoZ view, draw hits at the middle of the layer in the global Z coordinate, 
       // otherwise they won't align with 1D rechits, for which only one coordinate is known.
       if (wId.superLayer()==2) {
	 local[1]=0;
       } else {
	 local[0]=0;
       }
     }
   }

   geom->localToGlobal( rawid, local, global );
   pointSet->SetNextPoint( global[0], global[1], global[2] );
}

REGISTER_FWPROXYBUILDER( FWPSimHitProxyBuilder, PSimHit, "PSimHits", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
