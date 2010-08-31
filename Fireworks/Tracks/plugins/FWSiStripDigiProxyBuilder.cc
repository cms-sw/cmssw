// -*- C++ -*-
//
// Package:     Tracks
// Class  :     FWSiStripDigiProxyBuilder
//
//
// Original Author:
//         Created:  Thu Dec  6 18:01:21 PST 2007
// $Id: FWSiStripDigiProxyBuilder.cc,v 1.14 2010/08/23 15:26:42 yana Exp $
//

#include "TEveStraightLineSet.h"

#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/Tracks/interface/TrackUtils.h"

#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/DetId/interface/DetId.h"

class FWSiStripDigiProxyBuilder : public FWProxyBuilderBase
{
public:
  FWSiStripDigiProxyBuilder( void ) {}
  virtual ~FWSiStripDigiProxyBuilder( void ) {}

  REGISTER_PROXYBUILDER_METHODS();

private:
  virtual void build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* );
  FWSiStripDigiProxyBuilder( const FWSiStripDigiProxyBuilder& );    
  const FWSiStripDigiProxyBuilder& operator=( const FWSiStripDigiProxyBuilder& );
};

void
FWSiStripDigiProxyBuilder::build( const FWEventItem* iItem, TEveElementList* product, const FWViewContext* )
{
  const edm::DetSetVector<SiStripDigi>* digis = 0;

  iItem->get( digis );

  if( ! digis )
  {
    return;
  }
  const DetIdToMatrix* geom = iItem->getGeom();
   
  for( edm::DetSetVector<SiStripDigi>::const_iterator it = digis->begin(), end = digis->end();
       it != end; ++it )     
  { 
    edm::DetSet<SiStripDigi> ds = *it;
    const uint32_t& id = ds.id;

    const TGeoMatrix* matrix = geom->getMatrix( id );
    const float* pars = geom->getParameters( id );
        
    for( edm::DetSet<SiStripDigi>::const_iterator idigi = ds.data.begin(), idigiEnd = ds.data.end();
	 idigi != idigiEnd; ++idigi )        
    {
      TEveStraightLineSet *lineSet = new TEveStraightLineSet( "strip" );
      setupAddElement( lineSet, product );

      if( pars == 0 || (! matrix ))
      {
	fwLog( fwlog::kWarning ) 
	  << "failed get geometry and topology of SiStripDigi with detid: "
	  << id << std::endl;
	continue;
      }
      short strip = (*idigi).strip();
      double localTop[3] = { 0.0, 0.0, 0.0 };
      double localBottom[3] = { 0.0, 0.0, 0.0 };

      fireworks::localSiStrip( strip, localTop, localBottom, pars, id );

      double globalTop[3];
      double globalBottom[3];
      matrix->LocalToMaster( localTop, globalTop );
      matrix->LocalToMaster( localBottom, globalBottom );
  
      lineSet->AddLine( globalTop[0], globalTop[1], globalTop[2],
			globalBottom[0], globalBottom[1], globalBottom[2] );
    } // end of iteration over digis  
  } // end of iteration over the DetSetVector
}

REGISTER_FWPROXYBUILDER( FWSiStripDigiProxyBuilder, edm::DetSetVector<SiStripDigi>, "SiStripDigi", FWViewType::kAll3DBits | FWViewType::kRhoPhiBit | FWViewType::kRhoZBit );
