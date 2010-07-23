/*
 *  FWDTDigiProxyBuilder.cc
 *  FWorks
 *
 *  Created by Ianna Osborne on 6/7/10.
 *  Copyright 2010 FNAL. All rights reserved.
 *
 */

#include "TEveGeoNode.h"
#include "TEvePointSet.h"
#include "TEveCompound.h"
#include "TGeoArb8.h"
#include "TEveBox.h"

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
  void
  addTube( TEvePointSet* pointSet, const TGeoHMatrix* matrix, double localPos[3],  std::vector<TEveVector> &pars )
  {
    TEveBox* shape = new TEveBox( "DTube" );
    const Float_t vtx[24] = { localPos[0] - pars[0].fX / 2., -pars[0].fZ / 2., -pars[0].fY / 2.,
			      localPos[0] - pars[0].fX / 2.,  pars[0].fZ / 2., -pars[0].fY / 2.,
			      localPos[0] + pars[0].fX / 2.,  pars[0].fZ / 2., -pars[0].fY / 2.,
			      localPos[0] + pars[0].fX / 2., -pars[0].fZ / 2., -pars[0].fY / 2.,
			      localPos[0] - pars[0].fX / 2., -pars[0].fZ / 2.,  pars[0].fY / 2.,
			      localPos[0] - pars[0].fX / 2.,  pars[0].fZ / 2.,  pars[0].fY / 2.,
			      localPos[0] + pars[0].fX / 2.,  pars[0].fZ / 2.,  pars[0].fY / 2.,
			      localPos[0] + pars[0].fX / 2., -pars[0].fZ / 2.,  pars[0].fY / 2.};
    
    shape->SetVertices( vtx );
    shape->SetTransMatrix( *matrix );
    shape->SetMainTransparency( 75 );
    pointSet->AddElement( shape );
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
	
  virtual void buildViewType( const FWEventItem* iItem, TEveElementList* product, FWViewType::EType, const FWViewContext* );
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
    std::vector<TEveVector> pars = iItem->getGeom()->getPoints( layerId );

    int superLayer = layerId.superlayerId().superLayer();

    double localPos[3] = { 0.0, 0.0, 0.0 };
		
    const TGeoHMatrix* matrix = item()->getGeom()->getMatrix( layerId );
    
    const DTDigiCollection::Range &range = (*detIt).second;

    // Loop over the digis of this DetUnit
    for( DTDigiCollection::const_iterator digiIt = range.first;
	 digiIt != range.second; ++digiIt )
    {
      TEveCompound* compound = new TEveCompound( "DT digi compound" );
      compound->OpenCompound();
      product->AddElement( compound );
      if( pars.empty() ) {
	continue;
      }
			
      TEvePointSet* pointSet = new TEvePointSet();
      pointSet->SetMarkerStyle( 24 );
      pointSet->SetMarkerColor( 3 );
      compound->AddElement( pointSet );

      if( ! matrix ) 
      {
	fwLog( fwlog::kError ) << " failed get geometry of DT layer with detid: " 
			       << layerId << std::endl;
	continue;
      }
			
      int wire = (*digiIt).wire();

      // The x wire position in the layer, starting from its wire number.
      Float_t firstChannel = pars[1].fX;
      Float_t nChannels = pars[1].fZ;
      localPos[0] = ( wire - ( firstChannel - 1 ) - 0.5 ) * pars[0].fX - nChannels / 2.0 * pars[0].fX;

      if( type == FWViewType::k3D || type == FWViewType::kISpy )
      {
	addTube( pointSet, matrix, localPos, pars );
      }
      if(( type == FWViewType::kRhoPhi && superLayer != 2 ) ||
	 ( type == FWViewType::kRhoZ && superLayer == 2 ))
      {
	addMarkers( pointSet, matrix, localPos );
	addTube( pointSet, matrix, localPos, pars );
      }
    }		
  }
}

REGISTER_FWPROXYBUILDER( FWDTDigiProxyBuilder, DTDigiCollection, "DT Digis", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
