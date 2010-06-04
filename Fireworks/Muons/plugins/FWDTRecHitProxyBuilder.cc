// -*- C++ -*-
//
// Package:     Muons
// Class  :     FWDTRecHitProxyBuilder
//
// $Id: FWDTRecHitProxyBuilder.cc,v 1.5 2010/05/06 18:03:08 amraktad Exp $
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

	virtual void buildViewType(const DTRecHit1DPair& iData, unsigned int iIndex, TEveElement& oItemHolder, FWViewType::EType type , const FWViewContext*);
};

void
FWDTRecHitProxyBuilder::buildViewType( const DTRecHit1DPair& iData, unsigned int iIndex, TEveElement& oItemHolder, FWViewType::EType type, const FWViewContext* )
{
	int superLayer = iData.wireId().layerId().superlayerId().superLayer();

	// FIXME: These magic numbers should be gone 
	// as soon as we have 
	// access to proper geometry.
	// Note: radial thickness - almost constant about 5 cm
	
	float superLayerShift = 10.5;
	if( superLayer == 2 )
	{
		superLayerShift = -5.0;
	} 
	else if( superLayer == 3 ) 
	{
		superLayerShift = -10.5;
	}
	
	DTChamberId chamberId( iData.geographicalId() );
		
	const TGeoHMatrix* matrix = item()->getGeom()->getMatrix( chamberId );
    
   if( ! matrix ) 
   {
		fwLog( fwlog::kError ) << " failed get geometry of DT chamber with detid: " 
			<< chamberId << std::endl;
		return;
   }

   TEveStraightLineSet* recHitSet = new TEveStraightLineSet;
   setupAddElement( recHitSet, &oItemHolder );

	TEvePointSet* pointSet = new TEvePointSet();
	pointSet->SetMarkerSize( 2 );
	pointSet->SetMarkerStyle( 3 );
	setupAddElement( pointSet, &oItemHolder );
	 
   const DTRecHit1D* leftRecHit = iData.componentRecHit( Left );
   const DTRecHit1D* rightRecHit = iData.componentRecHit( Right );
	double lLocalPos[3] = { 0.0, 0.0, superLayerShift };
	double rLocalPos[3] = { 0.0, 0.0, superLayerShift };

	if( type == FWViewType::kRhoPhi && superLayer != 2 )
	{
		// FIXME: This position is valid only for RhoPhi View
		// and not for superLayer 2.
		lLocalPos[0] = leftRecHit->localPosition().x();
		rLocalPos[0] = rightRecHit->localPosition().x();
		addLineWithMarkers( recHitSet, pointSet, matrix, lLocalPos, rLocalPos );
	}
	else if( type == FWViewType::kRhoZ && superLayer == 2 )
	{
		// FIXME: This position is valid only for RhoZ View
		// and only for superLayer 2.
		lLocalPos[1] = -leftRecHit->localPosition().x();
		rLocalPos[1] = -rightRecHit->localPosition().x();
		addLineWithMarkers( recHitSet, pointSet, matrix, lLocalPos, rLocalPos );
	}
}

REGISTER_FWPROXYBUILDER( FWDTRecHitProxyBuilder, DTRecHit1DPair, "DT RecHits", FWViewType::kAllRPZBits );


