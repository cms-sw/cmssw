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

class FWDTDigiProxyBuilder : public FWProxyBuilderBase
{
public:
   FWDTDigiProxyBuilder( void ) {}
   virtual ~FWDTDigiProxyBuilder( void ) {}
	   
	REGISTER_PROXYBUILDER_METHODS();
	
private:
	// Disable default copy constructor
   FWDTDigiProxyBuilder( const FWDTDigiProxyBuilder& );
	// Disable default assignment operator
   const FWDTDigiProxyBuilder& operator=( const FWDTDigiProxyBuilder& );
	
	virtual void build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* );
};

void
FWDTDigiProxyBuilder::build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* )
{
	const DTDigiCollection* digis = 0;
	iItem->get( digis );
	
	if( ! digis )
	{
		fwLog( fwlog::kWarning ) << "ERROR: failed get DTDigis" << std::endl;
		return;
	}

	for( DTDigiCollection::DigiRangeIterator detIt = digis->begin(), detUnitItEnd = digis->end(); detIt != detUnitItEnd; ++detIt )
	{
		const DTLayerId& layerId = (*detIt).first;

		const TGeoHMatrix* matrix = item()->getGeom()->getMatrix( layerId );
		
		if( ! matrix ) 
		{
			fwLog( fwlog::kError ) << " failed get geometry of DT layer with detid: " 
			<< layerId << std::endl;
			return;
		}
		
		const DTDigiCollection::Range &range = (*detIt).second;

		// Loop over the digis of this DetUnit
		for( DTDigiCollection::const_iterator digiIt = range.first;
			 digiIt != range.second; ++digiIt )
		{
			TEveCompound* compound = new TEveCompound( "DT digi compound" );
			compound->OpenCompound();
			product->AddElement( compound );
			
			TEvePointSet* pointSet = new TEvePointSet();
			pointSet->SetMarkerSize(2);
			pointSet->SetMarkerStyle(2);
			compound->AddElement( pointSet );
			
			//int wire = (*digiIt).wire();
		}
	}
}

REGISTER_FWPROXYBUILDER( FWDTDigiProxyBuilder, DTDigiCollection, "DT Digis", FWViewType::kAllRPZBits );
