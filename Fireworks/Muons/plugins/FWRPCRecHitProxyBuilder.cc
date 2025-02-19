// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWRPCRecHitProxyBuilder
//
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: FWRPCRecHitProxyBuilder.cc,v 1.16 2010/11/11 20:25:28 amraktad Exp $
//

#include "TEveGeoNode.h"
#include "TEveStraightLineSet.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"

class FWRPCRecHitProxyBuilder : public FWSimpleProxyBuilderTemplate<RPCRecHit>
{
public:
   FWRPCRecHitProxyBuilder() {}
   virtual ~FWRPCRecHitProxyBuilder() {}
  
   virtual bool haveSingleProduct() const 
    { 
      return false; 
    }

   REGISTER_PROXYBUILDER_METHODS();

private:
  FWRPCRecHitProxyBuilder(const FWRPCRecHitProxyBuilder&);
  const FWRPCRecHitProxyBuilder& operator=(const FWRPCRecHitProxyBuilder&); 
 
  virtual void buildViewType(const RPCRecHit& iData, 
                             unsigned int iIndex, 
                             TEveElement& oItemHolder, 
                             FWViewType::EType type, 
                             const FWViewContext*);
};


void
FWRPCRecHitProxyBuilder::buildViewType(const RPCRecHit& iData,
                                       unsigned int iIndex, 
                                       TEveElement& oItemHolder, 
                                       FWViewType::EType type,
                                       const FWViewContext*)
{
  RPCDetId rpcId = iData.rpcId();
  unsigned int rawid = rpcId.rawId();
  
  const FWGeometry *geom = item()->getGeom();

  if( ! geom->contains( rawid ))
  {
    fwLog( fwlog::kError ) << "failed to get geometry of RPC roll with detid: " 
			   << rawid <<std::endl;
    return;
  }

  TEveStraightLineSet* recHitSet = new TEveStraightLineSet;
  recHitSet->SetLineWidth(3);

  if( type == FWViewType::k3D || type == FWViewType::kISpy ) 
  {
    TEveGeoShape* shape = geom->getEveShape( rawid );
    shape->SetMainTransparency( 75 );
    shape->SetMainColor( item()->defaultDisplayProperties().color());
    recHitSet->AddElement( shape );
  }

  float localX = iData.localPosition().x();
  float localY = iData.localPosition().y();
  float localZ = iData.localPosition().z();
  
  float localXerr = sqrt(iData.localPositionError().xx());
  float localYerr = sqrt(iData.localPositionError().yy());

  float localU1[3] = 
    {
      localX - localXerr, localY, localZ 
    };
  
  float localU2[3] = 
    {
      localX + localXerr, localY, localZ 
    };
  
  float localV1[3] = 
    {
      localX, localY - localYerr, localZ
    };
  
  float localV2[3] = 
    {
      localX, localY + localYerr, localZ
    };

  float globalU1[3];
  float globalU2[3];
  float globalV1[3];
  float globalV2[3];

  FWGeometry::IdToInfoItr det = geom->find( rawid );
 
  geom->localToGlobal( *det, localU1, globalU1 );
  geom->localToGlobal( *det, localU2, globalU2 );
  geom->localToGlobal( *det, localV1, globalV1 );
  geom->localToGlobal( *det, localV2, globalV2 );
 
  recHitSet->AddLine( globalU1[0], globalU1[1], globalU1[2],
                      globalU2[0], globalU2[1], globalU2[2] );

  recHitSet->AddLine( globalV1[0], globalV1[1], globalV1[2],
                      globalV2[0], globalV2[1], globalV2[2] );

  setupAddElement( recHitSet, &oItemHolder );
}

REGISTER_FWPROXYBUILDER( FWRPCRecHitProxyBuilder, RPCRecHit, "RPC RecHits", 
                         FWViewType::kAll3DBits | FWViewType::kAllRPZBits);
