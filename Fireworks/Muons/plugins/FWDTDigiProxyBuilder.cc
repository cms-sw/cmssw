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
  addTube( TEveBox* shape, const TGeoHMatrix* matrix, double localPos[3],  std::vector<Float_t> &pars )
  {
    const Float_t width = pars[0] / 2.;
    const Float_t thickness = pars[1] / 2.;
    const Float_t length = pars[2] / 2.;
    
    const Float_t vtx[24] = { localPos[0] - width, -length, -thickness,
			      localPos[0] - width,  length, -thickness,
			      localPos[0] + width,  length, -thickness,
			      localPos[0] + width, -length, -thickness,
			      localPos[0] - width, -length,  thickness,
			      localPos[0] - width,  length,  thickness,
			      localPos[0] + width,  length,  thickness,
			      localPos[0] + width, -length,  thickness };
    
    shape->SetVertices( vtx );
    shape->SetTransMatrix( *matrix );
    shape->SetMainTransparency( 75 );
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
  const DTDigiCollection* digis = 0;
  iItem->get( digis );
	
  if( ! digis )
  {
    return;
  }
	
  for( DTDigiCollection::DigiRangeIterator det = digis->begin(), detEnd = digis->end(); det != detEnd; ++det )
  {
    const DTLayerId& layerId = (*det).first;
    std::vector<Float_t> pars = iItem->getGeom()->getParameters( layerId );

    int superLayer = layerId.superlayerId().superLayer();

    double localPos[3] = { 0.0, 0.0, 0.0 };
		
    const TGeoHMatrix* matrix = item()->getGeom()->getMatrix( layerId );
    
    const DTDigiCollection::Range &range = (*det).second;

    // Loop over the digis of this DetUnit
    for( DTDigiCollection::const_iterator it = range.first;
	 it != range.second; ++it )
    {
      TEveCompound* compound = createCompound();
      setupAddElement( compound, product );

      if( pars.empty() || (! matrix )) {
	fwLog( fwlog::kError )
	  << "failed get geometry of DT layer with detid: " 
	  << layerId << std::endl;

	continue;
      }
			
      // The x wire position in the layer, starting from its wire number.
      Float_t firstChannel = pars[3];
      Float_t nChannels = pars[5];
      localPos[0] = ((*it).wire() - ( firstChannel - 1 ) - 0.5 ) * pars[0] - nChannels / 2.0 * pars[0];

      if( type == FWViewType::k3D || type == FWViewType::kISpy )
      {
	TEveBox* box = new TEveBox;
	setupAddElement( box, compound );
	addTube( box, matrix, localPos, pars );
      }
      if(( type == FWViewType::kRhoPhi && superLayer != 2 ) ||
	 ( type == FWViewType::kRhoZ && superLayer == 2 ))
      {
	TEvePointSet* pointSet = new TEvePointSet;
	pointSet->SetMarkerStyle( 24 );
	setupAddElement( pointSet, compound );	
	addMarkers( pointSet, matrix, localPos );

	TEveBox* box = new TEveBox;
	setupAddElement( box, compound );
	addTube( box, matrix, localPos, pars );
      }
    }		
  }
}

REGISTER_FWPROXYBUILDER( FWDTDigiProxyBuilder, DTDigiCollection, "DT Digis", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
