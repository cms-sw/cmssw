// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWRPCRecHitProxyBuilder
//
//
// Original Author:
//         Created:  Sun Jan  6 23:42:33 EST 2008
// $Id: FWRPCRecHitProxyBuilder.cc,v 1.8 2010/06/30 14:37:07 mccauley Exp $
//

#include "TEveGeoNode.h"
#include "TEveStraightLineSet.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"

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

  const TGeoHMatrix* matrix = item()->getGeom()->getMatrix(rpcId);
  
  if ( ! matrix ) 
  {
    std::cout << "ERROR: failed get geometry of RPC reference volume with detid: "
              << rpcId << std::endl;
    return;
  }
 
  std::stringstream s;
  s << "rec hit" << iIndex;

  TEveStraightLineSet* recHitSet = new TEveStraightLineSet(s.str().c_str());
  recHitSet->SetLineWidth(3);

  /*

  NOTE: Do not draw shape until geometry bug is fixed.

  TEveGeoShape* shape = item()->getGeom()->getShape(rpcId);

  if ( shape && ( type == FWViewType::k3D || type == FWViewType::kISpy ) ) 
  {
    shape->SetMainTransparency(75);
    shape->SetMainColor(item()->defaultDisplayProperties().color());
    recHitSet->AddElement(shape);
  }
  */

  double localX = iData.localPosition().x();
  double localY = iData.localPosition().y();
  double localZ = iData.localPosition().z();
  
  double localXerr = sqrt(iData.localPositionError().xx());
  double localYerr = sqrt(iData.localPositionError().yy());

  double localU1[3] = 
    {
      localX - localXerr, localY, localZ 
    };
  
  double localU2[3] = 
    {
      localX + localXerr, localY, localZ 
    };
  
  double localV1[3] = 
    {
      localX, localY - localYerr, localZ
    };
  
  double localV2[3] = 
    {
      localX, localY + localYerr, localZ
    };

  double globalU1[3];
  double globalU2[3];
  double globalV1[3];
  double globalV2[3];
 
  matrix->LocalToMaster(localU1, globalU1);
  matrix->LocalToMaster(localU2, globalU2);
  matrix->LocalToMaster(localV1, globalV1);
  matrix->LocalToMaster(localV2, globalV2);
 
  recHitSet->AddLine(globalU1[0], globalU1[1], globalU1[2],
                     globalU2[0], globalU2[1], globalU2[2]);

  recHitSet->AddLine(globalV1[0], globalV1[1], globalV1[2],
                     globalV2[0], globalV2[1], globalV2[2]);

  setupAddElement(recHitSet, &oItemHolder);
}

REGISTER_FWPROXYBUILDER( FWRPCRecHitProxyBuilder, RPCRecHit, "RPC RecHits", FWViewType::kAll3DBits | FWViewType::kRhoPhiBit  | FWViewType::kRhoZBit);
