// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWCSCRecHitProxyBuilder
//
// $Id: FWCSCRecHitProxyBuilder.cc,v 1.7 2010/06/15 17:05:59 mccauley Exp $
//

#include "TEveStraightLineSet.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"

class FWCSCRecHitProxyBuilder : public FWSimpleProxyBuilderTemplate<CSCRecHit2D>
{
public:
   FWCSCRecHitProxyBuilder() {}
   virtual ~FWCSCRecHitProxyBuilder() {}
  
   REGISTER_PROXYBUILDER_METHODS();

private:
   FWCSCRecHitProxyBuilder(const FWCSCRecHitProxyBuilder&);
   const FWCSCRecHitProxyBuilder& operator=(const FWCSCRecHitProxyBuilder&);

   void build(const CSCRecHit2D& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*);
};

void
FWCSCRecHitProxyBuilder::build(const CSCRecHit2D& iData,           
                               unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext*)
{       
  const TGeoHMatrix* matrix = item()->getGeom()->getMatrix(iData.cscDetId().rawId());
  
  if ( ! matrix ) 
  {     
     fwLog(fwlog::kError) << " failed to get geometry of CSC layer with detid: " 
                          << iData.cscDetId().rawId() <<std::endl;
    return;
  }

  std::stringstream s;
  s << "layer" << iIndex;

  TEveStraightLineSet* recHitSet = new TEveStraightLineSet(s.str().c_str());
  recHitSet->SetLineWidth(3);
  setupAddElement(recHitSet, &oItemHolder);
  
  double localPositionX = iData.localPosition().x();
  double localPositionY = iData.localPosition().y();
  
  double localPositionXX = sqrt(iData.localPositionError().xx());
  double localPositionYY = sqrt(iData.localPositionError().yy());
  
  double localU1Point[3] = 
    {
      localPositionX - localPositionXX, localPositionY, 0.0
    };
  
  double localU2Point[3] = 
    {
      localPositionX + localPositionXX, localPositionY, 0.0
    };
  
  double localV1Point[3] = 
    {
      localPositionX, localPositionY - localPositionYY, 0.0
    };
  
  double localV2Point[3] = 
    {
      localPositionX, localPositionY + localPositionYY, 0.0
    };

  double globalU1Point[3];
  double globalU2Point[3];
  double globalV1Point[3];
  double globalV2Point[3];

  matrix->LocalToMaster(localU1Point, globalU1Point);
  matrix->LocalToMaster(localU2Point, globalU2Point);
  matrix->LocalToMaster(localV1Point, globalV1Point);
  matrix->LocalToMaster(localV2Point, globalV2Point);
 
  recHitSet->AddLine(globalU1Point[0], globalU1Point[1], globalU1Point[2], 
                     globalU2Point[0], globalU2Point[1], globalU2Point[2]);
 
  recHitSet->AddLine(globalV1Point[0], globalV1Point[1], globalV1Point[2], 
                     globalV2Point[0], globalV2Point[1], globalV2Point[2]);
}

REGISTER_FWPROXYBUILDER( FWCSCRecHitProxyBuilder, CSCRecHit2D, "CSC RecHits", FWViewType::kAll3DBits );


