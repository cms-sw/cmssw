// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWDTRecHitProxyBuilder
//
// $Id: FWDTRecHitProxyBuilder.cc,v 1.13 2010/11/11 20:25:28 amraktad Exp $
//

#include "TEvePointSet.h"
#include "TEveStraightLineSet.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"

using namespace DTEnums;

class FWDTRecHitProxyBuilder : public FWSimpleProxyBuilderTemplate<DTRecHit1DPair>
{
public:
  FWDTRecHitProxyBuilder( void ) {}
  virtual ~FWDTRecHitProxyBuilder( void ) {}
	
  virtual bool haveSingleProduct() const { return false; }
   
  REGISTER_PROXYBUILDER_METHODS();

private:
  // Disable default copy constructor
  FWDTRecHitProxyBuilder( const FWDTRecHitProxyBuilder& );
  // Disable default assignment operator
  const FWDTRecHitProxyBuilder& operator=( const FWDTRecHitProxyBuilder& );

  virtual void buildViewType( const DTRecHit1DPair& iData, unsigned int iIndex, TEveElement& oItemHolder, FWViewType::EType type , const FWViewContext* );
};

void
FWDTRecHitProxyBuilder::buildViewType( const DTRecHit1DPair& iData, unsigned int iIndex, TEveElement& oItemHolder, FWViewType::EType type, const FWViewContext* )
{
  const DTLayerId& layerId = iData.wireId().layerId();
  int superLayer = layerId.superlayerId().superLayer();

  const FWGeometry *geom = item()->getGeom();

  if( ! geom->contains( layerId ))
  {
    fwLog( fwlog::kError ) << "failed get geometry of DT layer with detid: " 
			   << layerId << std::endl;
    return;
  }

  TEveStraightLineSet* recHitSet = new TEveStraightLineSet;
  setupAddElement( recHitSet, &oItemHolder );

  TEvePointSet* pointSet = new TEvePointSet;
  setupAddElement( pointSet, &oItemHolder );
	 
  const DTRecHit1D* leftRecHit = iData.componentRecHit( Left );
  const DTRecHit1D* rightRecHit = iData.componentRecHit( Right );
  float lLocalPos[3] = { leftRecHit->localPosition().x(), 0.0, 0.0 };
  float rLocalPos[3] = { rightRecHit->localPosition().x(), 0.0, 0.0 };

  if(( (type == FWViewType::kRhoPhi || type == FWViewType::kRhoPhiPF) && superLayer != 2 ) ||
    ( type == FWViewType::kRhoZ && superLayer == 2 ) ||
     type == FWViewType::k3D ||
     type == FWViewType::kISpy )
  {
    float leftGlobalPoint[3];
    float rightGlobalPoint[3];
		
    geom->localToGlobal( layerId, lLocalPos, leftGlobalPoint, rLocalPos, rightGlobalPoint );
		
    pointSet->SetNextPoint( leftGlobalPoint[0],  leftGlobalPoint[1],  leftGlobalPoint[2] ); 
    pointSet->SetNextPoint( rightGlobalPoint[0], rightGlobalPoint[1], rightGlobalPoint[2] );
		
    recHitSet->AddLine( leftGlobalPoint[0],  leftGlobalPoint[1],  leftGlobalPoint[2], 
			rightGlobalPoint[0], rightGlobalPoint[1], rightGlobalPoint[2] );		
  }
}

REGISTER_FWPROXYBUILDER( FWDTRecHitProxyBuilder, DTRecHit1DPair, "DT RecHits",  FWViewType::kAll3DBits | FWViewType::kAllRPZBits );


