// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWRPCRecHitProxyBuilder
//
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: FWRPCRecHitProxyBuilder.cc,v 1.3 2010/04/16 16:40:13 yana Exp $
//

#include "TEveGeoNode.h"
#include "TEvePointSet.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"

#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"

class FWRPCRecHitProxyBuilder : public FWSimpleProxyBuilderTemplate<RPCRecHit>
{
public:
   FWRPCRecHitProxyBuilder() {}
   virtual ~FWRPCRecHitProxyBuilder() {}
  
   REGISTER_PROXYBUILDER_METHODS();

private:
  FWRPCRecHitProxyBuilder(const FWRPCRecHitProxyBuilder&);
  const FWRPCRecHitProxyBuilder& operator=(const FWRPCRecHitProxyBuilder&); 
  
  void build(const RPCRecHit& iData,
             unsigned int iIndex, TEveElement& oItemHolder);
};

void
FWRPCRecHitProxyBuilder::build(const RPCRecHit& iData,
                               unsigned int iIndex, TEveElement& oItemHolder)
{
  RPCDetId rpcId = iData.rpcId();

  const TGeoHMatrix* matrix = item()->getGeom()->getMatrix(rpcId);
  
  if ( ! matrix ) 
  {
    std::cout << "ERROR: failed get geometry of RPC reference volume with detid: "
              << rpcId << std::endl;
    return;
  }
 
  TEveGeoShape* shape = item()->getGeom()->getShape(rpcId);

  double localPoint[3];
  double globalPoint[3];

  localPoint[0] = iData.localPosition().x();
  localPoint[1] = iData.localPosition().y();
  localPoint[2] = iData.localPosition().z();
	 
  TEvePointSet* pointSet = new TEvePointSet;
  pointSet->SetMarkerStyle(2);
  pointSet->SetMarkerSize(3);
  setupAddElement(pointSet, &oItemHolder);

  if ( shape ) 
  {
    shape->SetMainTransparency(75);
    shape->SetMainColor(item()->defaultDisplayProperties().color());
    pointSet->AddElement(shape);
  }

  if( rpcId.layer() == 1 && rpcId.station() < 3 )
    localPoint[0] = -localPoint[0];
	
  matrix->LocalToMaster(localPoint, globalPoint);

  pointSet->SetNextPoint(globalPoint[0], globalPoint[1], globalPoint[2]);
}

REGISTER_FWPROXYBUILDER( FWRPCRecHitProxyBuilder, RPCRecHit, "RPC RecHits", FWViewType::kAll3DBits | FWViewType::kRhoPhiBit  | FWViewType::kRhoZBit);
