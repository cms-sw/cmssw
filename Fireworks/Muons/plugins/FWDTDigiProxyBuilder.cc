/*
 *  FWDTDigiProxyBuilder.cc
 *  FWorks
 *
 *  Created by Ianna Osborne on 6/7/10.
 *  Copyright 2010 FNAL. All rights reserved.
 *
 */

#include "TEvePointSet.h"
#include "TEveCompound.h"

#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"

namespace 
{
	void 
	addMarkers( TEvePointSet* pointSet, const TGeoHMatrix* matrix, double localPos[3] ) 
	{
		double globalPos[3];			
		matrix->LocalToMaster( localPos, globalPos );
		pointSet->SetNextPoint( globalPos[0],  globalPos[1],  globalPos[2] );
	}
}

class FWDTDigiProxyBuilder : public FWProxyBuilderBase
{
public:
   FWDTDigiProxyBuilder( void ) {}
   virtual ~FWDTDigiProxyBuilder( void ) {}

	virtual bool haveSingleProduct( void ) const { return false; }
	   
	REGISTER_PROXYBUILDER_METHODS();
	
private:
	// Disable default copy constructor
   FWDTDigiProxyBuilder( const FWDTDigiProxyBuilder& );
	// Disable default assignment operator
   const FWDTDigiProxyBuilder& operator=( const FWDTDigiProxyBuilder& );
	
   virtual void buildViewType(const FWEventItem* iItem, TEveElementList* product, FWViewType::EType, const FWViewContext*);
};

void
FWDTDigiProxyBuilder::buildViewType( const FWEventItem* iItem, TEveElementList* product, FWViewType::EType type, const FWViewContext* )
{
	// FIXME: This colour does not have any affect on how the points look like.
	product->SetMainColor( iItem->defaultDisplayProperties().color() );

	const DTDigiCollection* digis = 0;
	iItem->get( digis );
	
	if( ! digis )
	{
		fwLog( fwlog::kWarning ) << "WARNING: failed get DTDigis" << std::endl;
		return;
	}
	
	for( DTDigiCollection::DigiRangeIterator detIt = digis->begin(), detUnitItEnd = digis->end(); detIt != detUnitItEnd; ++detIt )
	{
		const DTLayerId& layerId = (*detIt).first;
		int layer = layerId.layer();
		int superLayer = layerId.superlayerId().superLayer();
		DTChamberId chamberId = layerId.superlayerId().chamberId();
		
		float superLayerShift = 15.0;
		// Sample superlayer:
		// x/2 = 1063.2;
		// y/2 = 1255.5;
		// z/2 = 26.75;
		
		if( superLayer == 2 )
		{
			superLayerShift = 0.0;
		} 
		else if( superLayer == 3 ) 
		{
			superLayerShift = -5.35;
		}
		// The distance between the wire planes:
		float layerShift = 1.3375;
		float wireShift = 4.2;
		
		double localPos[3] = { 0.0, 0.0, 0.0 };
		
		const TGeoHMatrix* matrix = item()->getGeom()->getMatrix( chamberId );
				
		const DTDigiCollection::Range &range = (*detIt).second;

		// Loop over the digis of this DetUnit
		for( DTDigiCollection::const_iterator digiIt = range.first;
			 digiIt != range.second; ++digiIt )
		{
			TEveCompound* compound = new TEveCompound( "DT digi compound" );
			compound->OpenCompound();
			product->AddElement( compound );

			TEvePointSet* pointSet = new TEvePointSet();
			pointSet->SetMarkerStyle( 24 );
			pointSet->SetMarkerColor( 3 );
			compound->AddElement( pointSet );
			
			if( ! matrix ) 
			{
				fwLog( fwlog::kError ) << " failed get geometry of DT chamber with detid: " 
				<< layerId << std::endl;
				continue;
			}
			
			int wire = (*digiIt).wire();
			// FIXME: We need to correct the wire position:
			// The x wire position in the layer, starting from its wire number.
			// float wPos = ( wire - ( firstWire - 1 ) - 0.5 ) * 4.2 - nChannels / 2.0 * 4.2;

			localPos[2] = superLayerShift - layer * layerShift;
			
			if( type == FWViewType::kRhoPhi && superLayer != 2 )
			{
				// The center is in the middle of DT chamber.
				// We need to offset it by half/width.
				localPos[0] = - 106.2 + wire * wireShift;
				addMarkers( pointSet, matrix, localPos );
			}
			else if( type == FWViewType::kRhoZ && superLayer == 2 )
			{
				// The center is in the middle of DT chamber.
				// We need to offset it by half/length.
				localPos[1] = 106.2 - ( wire + 1 ) * wireShift;
				addMarkers( pointSet, matrix, localPos );
			}
		}		
	}
}

REGISTER_FWPROXYBUILDER( FWDTDigiProxyBuilder, DTDigiCollection, "DT Digis", FWViewType::kAllRPZBits );
