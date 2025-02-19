// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWCSCRecHitProxyBuilder
//
// $Id: FWCSCRecHitProxyBuilder.cc,v 1.13 2010/09/07 15:46:48 yana Exp $
//

#include "TEvePointSet.h"
#include "TEveStraightLineSet.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"

class FWCSCRecHitProxyBuilder : public FWSimpleProxyBuilderTemplate<CSCRecHit2D>
{
public:
  FWCSCRecHitProxyBuilder( void ) {}
  virtual ~FWCSCRecHitProxyBuilder( void ) {}
  
  REGISTER_PROXYBUILDER_METHODS();

private:
  FWCSCRecHitProxyBuilder( const FWCSCRecHitProxyBuilder& );
  const FWCSCRecHitProxyBuilder& operator=( const FWCSCRecHitProxyBuilder& );

  void build( const CSCRecHit2D& iData, unsigned int iIndex, TEveElement& oItemHolder, const FWViewContext* );
};

void
FWCSCRecHitProxyBuilder::build( const CSCRecHit2D& iData,           
				unsigned int iIndex, TEveElement& oItemHolder, 
                                const FWViewContext* )
{       
  const FWGeometry *geom = item()->getGeom();
  unsigned int rawid = iData.cscDetId().rawId();
  
  if( ! geom->contains( rawid ))
  {
    fwLog( fwlog::kError ) << "failed to get geometry of CSC layer with detid: " 
			   << rawid <<std::endl;
    return;
  }
  FWGeometry::IdToInfoItr det = geom->find( rawid );

  TEveStraightLineSet* recHitSet = new TEveStraightLineSet;
  setupAddElement( recHitSet, &oItemHolder );

  TEvePointSet* pointSet = new TEvePointSet;
  setupAddElement( pointSet, &oItemHolder );
  
  float localPositionX = iData.localPosition().x();
  float localPositionY = iData.localPosition().y();
  
  float localPositionXX = sqrt( iData.localPositionError().xx());
  float localPositionYY = sqrt( iData.localPositionError().yy());
  
  float localU1Point[3] = 
    {
      localPositionX - localPositionXX, localPositionY, 0.0
    };
  
  float localU2Point[3] = 
    {
      localPositionX + localPositionXX, localPositionY, 0.0
    };
  
  float localV1Point[3] = 
    {
      localPositionX, localPositionY - localPositionYY, 0.0
    };
  
  float localV2Point[3] = 
    {
      localPositionX, localPositionY + localPositionYY, 0.0
    };

  float globalU1Point[3];
  float globalU2Point[3];
  float globalV1Point[3];
  float globalV2Point[3];

  geom->localToGlobal( *det, localU1Point, globalU1Point );
  geom->localToGlobal( *det, localU2Point, globalU2Point );
  geom->localToGlobal( *det, localV1Point, globalV1Point );
  geom->localToGlobal( *det, localV2Point, globalV2Point );

  pointSet->SetNextPoint( globalU1Point[0], globalU1Point[1], globalU1Point[2] ); 
  pointSet->SetNextPoint( globalU2Point[0], globalU2Point[1], globalU2Point[2] );
  pointSet->SetNextPoint( globalV1Point[0], globalV1Point[1], globalV1Point[2] );
  pointSet->SetNextPoint( globalV2Point[0], globalV2Point[1], globalV2Point[2] );
  
  recHitSet->AddLine( globalU1Point[0], globalU1Point[1], globalU1Point[2], 
                      globalU2Point[0], globalU2Point[1], globalU2Point[2] );
 
  recHitSet->AddLine( globalV1Point[0], globalV1Point[1], globalV1Point[2], 
                      globalV2Point[0], globalV2Point[1], globalV2Point[2] );
}

REGISTER_FWPROXYBUILDER( FWCSCRecHitProxyBuilder, CSCRecHit2D, "CSC RecHits", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );


