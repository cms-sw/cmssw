// -*- C++ -*-
//
// Package:     Tracks
// Class  :     FWSiPixelDigiProxyBuilder
//
//
// Original Author:
//         Created:  Thu Dec  6 18:01:21 PST 2007
// $Id: FWSiPixelDigiProxyBuilder.cc,v 1.10 2010/06/18 12:44:47 yana Exp $
//

#include "TEveCompound.h"
#include "TEvePointSet.h"

#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"

#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/Common/interface/DetSetVector.h"

class FWSiPixelDigiProxyBuilder : public FWProxyBuilderBase
{
public:
  FWSiPixelDigiProxyBuilder( void ) {}
  virtual ~FWSiPixelDigiProxyBuilder( void ) {}

  REGISTER_PROXYBUILDER_METHODS();

private:
  // Disable default copy constructor
  FWSiPixelDigiProxyBuilder( const FWSiPixelDigiProxyBuilder& );    
  // Disable default assignment operator
  const FWSiPixelDigiProxyBuilder& operator=( const FWSiPixelDigiProxyBuilder& );

  virtual void build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* );
};

void FWSiPixelDigiProxyBuilder::build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* )
{
  const edm::DetSetVector<PixelDigi>* digis = 0;
  iItem->get( digis );

  if( ! digis )
  {
    return;
  }
  
  for( edm::DetSetVector<PixelDigi>::const_iterator it = digis->begin(), end = digis->end();
        it != end; ++it )
  {
    edm::DetSet<PixelDigi> ds = *it;
    unsigned int id = ds.id;

    const DetIdToMatrix *geom = iItem->getGeom();
    const TGeoHMatrix *matrix = geom->getMatrix( id );
    std::vector<Float_t> pars = geom->getParameters( id );
         
    for( edm::DetSet<PixelDigi>::const_iterator idigi = ds.data.begin(), idigiEnd = ds.data.end();
	 idigi != idigiEnd; ++idigi )
    {
      TEvePointSet* pointSet = new TEvePointSet;
      pointSet->SetMarkerSize( 2 );
      pointSet->SetMarkerStyle( 2 );
      pointSet->SetMarkerColor( 46 );
      setupAddElement( pointSet, product );

      if( ! matrix || pars.empty()) 
      {
	fwLog( fwlog::kWarning ) 
	  << "failed get geometry of SiPixelDigi with detid: "
	  << id << std::endl;
      }
      else
      {	
	double localPoint[3] = {     
	  fireworks::pixelLocalX(( *idigi ).row(), pars[0] ),
	  fireworks::pixelLocalY(( *idigi ).column(), pars[1] ),
	  0.0 };
	
	double globalPoint[3];
        
	matrix->LocalToMaster( localPoint, globalPoint );
      
	pointSet->SetNextPoint( globalPoint[0], globalPoint[1], globalPoint[2] );
      }
    } // end of iteration over digis in range   
  } // end of iteration over the DetSetVector
}


REGISTER_FWPROXYBUILDER( FWSiPixelDigiProxyBuilder, edm::DetSetVector<PixelDigi>, "SiPixelDigi", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
