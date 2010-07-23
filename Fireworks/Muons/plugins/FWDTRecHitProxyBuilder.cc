// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWDTRecHitProxyBuilder
//
// $Id: FWDTRecHitProxyBuilder.cc,v 1.7 2010/06/10 14:01:37 yana Exp $
//

#include "TEvePointSet.h"
#include "TEveStraightLineSet.h"

#include "Fireworks/Core/interface/FWSimpleProxyBuilderTemplate.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"

namespace 
{
  void 
  addLineWithMarkers( TEveStraightLineSet* recHitSet, TEvePointSet* pointSet, 
		      const TGeoHMatrix* matrix, double lLocalPos[3], double rLocalPos[3] ) 
  {
    double leftGlobalPoint[3];
    double rightGlobalPoint[3];
		
    matrix->LocalToMaster( lLocalPos, leftGlobalPoint );
    matrix->LocalToMaster( rLocalPos, rightGlobalPoint );
		
    pointSet->SetNextPoint( leftGlobalPoint[0],  leftGlobalPoint[1],  leftGlobalPoint[2] ); 
    pointSet->SetNextPoint( rightGlobalPoint[0], rightGlobalPoint[1], rightGlobalPoint[2] );
		
    recHitSet->AddLine( leftGlobalPoint[0],  leftGlobalPoint[1],  leftGlobalPoint[2], 
			rightGlobalPoint[0], rightGlobalPoint[1], rightGlobalPoint[2] );		
  }
}

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

  const TGeoHMatrix* matrix = item()->getGeom()->getMatrix( layerId );
    
  if( ! matrix ) 
  {
    fwLog( fwlog::kError ) << " failed get geometry of DT layer with detid: " 
			   << layerId << std::endl;
    return;
  }

  TEveStraightLineSet* recHitSet = new TEveStraightLineSet;
  setupAddElement( recHitSet, &oItemHolder );

  TEvePointSet* pointSet = new TEvePointSet();
  setupAddElement( pointSet, &oItemHolder );
	 
  const DTRecHit1D* leftRecHit = iData.componentRecHit( Left );
  const DTRecHit1D* rightRecHit = iData.componentRecHit( Right );
  double lLocalPos[3] = { leftRecHit->localPosition().x(), 0.0, 0.0 };
  double rLocalPos[3] = { rightRecHit->localPosition().x(), 0.0, 0.0 };

  if(( type == FWViewType::kRhoPhi && superLayer != 2 ) ||
    ( type == FWViewType::kRhoZ && superLayer == 2 ) ||
     type == FWViewType::k3D ||
     type == FWViewType::kISpy )
  {
    addLineWithMarkers( recHitSet, pointSet, matrix, lLocalPos, rLocalPos );
  }
}

REGISTER_FWPROXYBUILDER( FWDTRecHitProxyBuilder, DTRecHit1DPair, "DT RecHits",  FWViewType::kAll3DBits | FWViewType::kAllRPZBits );


