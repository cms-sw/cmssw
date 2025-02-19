// -*- C++ -*-
//
// Package:     Tracks
// Class  :     FWSiStripDigiProxyBuilder
//
//
// Original Author:
//         Created:  Thu Dec  6 18:01:21 PST 2007
// $Id: FWSiStripDigiProxyBuilder.cc,v 1.18 2010/09/07 15:46:49 yana Exp $
//

#include "TEveStraightLineSet.h"

#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGeometry.h"
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
  const FWGeometry* geom = iItem->getGeom();
   
  for( edm::DetSetVector<SiStripDigi>::const_iterator it = digis->begin(), end = digis->end();
       it != end; ++it )     
  { 
    edm::DetSet<SiStripDigi> ds = *it;
    const uint32_t& id = ds.id;

    const float* pars = geom->getParameters( id );
        
    for( edm::DetSet<SiStripDigi>::const_iterator idigi = ds.data.begin(), idigiEnd = ds.data.end();
	 idigi != idigiEnd; ++idigi )        
    {
      TEveStraightLineSet *lineSet = new TEveStraightLineSet;
      setupAddElement( lineSet, product );

      if( ! geom->contains( id ))
      {
	fwLog( fwlog::kWarning ) 
	  << "failed get geometry and topology of SiStripDigi with detid: "
	  << id << std::endl;
	continue;
      }
      float localTop[3] = { 0.0, 0.0, 0.0 };
      float localBottom[3] = { 0.0, 0.0, 0.0 };

      fireworks::localSiStrip(( *idigi ).strip(), localTop, localBottom, pars, id );

      float globalTop[3];
      float globalBottom[3];
      geom->localToGlobal( id, localTop, globalTop, localBottom, globalBottom );
  
      lineSet->AddLine( globalTop[0], globalTop[1], globalTop[2],
			globalBottom[0], globalBottom[1], globalBottom[2] );
    } // end of iteration over digis  
  } // end of iteration over the DetSetVector
}

REGISTER_FWPROXYBUILDER( FWSiStripDigiProxyBuilder, edm::DetSetVector<SiStripDigi>, "SiStripDigi", FWViewType::kAll3DBits | FWViewType::kAllRPZBits );
