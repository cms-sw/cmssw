#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerSubStrctBuilder.h"

#include <bitset>

CmsTrackerBuilder::CmsTrackerBuilder( unsigned int totalBlade )
  : m_totalBlade( totalBlade )
{}

void
CmsTrackerBuilder::buildComponent( DDFilteredView& fv, GeometricDet* g, std::string s )
{
  CmsTrackerSubStrctBuilder theCmsTrackerSubStrctBuilder( m_totalBlade );

  GeometricDet* subdet = new GeometricDet( &fv, theCmsTrackerStringToEnum.type( ExtractStringFromDDD::getString( s, &fv )));
  switch( theCmsTrackerStringToEnum.type( ExtractStringFromDDD::getString( s, &fv )))
  {
  case GeometricDet::PixelBarrel:
    theCmsTrackerSubStrctBuilder.build( fv, subdet, s );      
    break;
  case GeometricDet::PixelEndCap:
    theCmsTrackerSubStrctBuilder.build( fv, subdet, s );      
    break;
  case GeometricDet::TIB:
    theCmsTrackerSubStrctBuilder.build( fv, subdet, s );      
    break;
  case GeometricDet::TOB:
    theCmsTrackerSubStrctBuilder.build( fv, subdet, s );    
    break;
  case GeometricDet::TEC:
    theCmsTrackerSubStrctBuilder.build( fv, subdet, s );      
    break;
  case GeometricDet::TID:
    theCmsTrackerSubStrctBuilder.build( fv, subdet, s );      
    break;
  default:
    edm::LogError( "CmsTrackerBuilder" ) << " ERROR - I was expecting a SubDet, I got a " << ExtractStringFromDDD::getString( s, &fv );
  }
  
  g->addComponent( subdet );
}

void
CmsTrackerBuilder::sortNS( DDFilteredView& fv, GeometricDet* det )
{  
  GeometricDet::ConstGeometricDetContainer & comp = det->components();
  std::stable_sort( comp.begin(), comp.end(), subDetByType());
  
  for( uint32_t i = 0; i < comp.size(); i++ )
  {
    uint32_t temp= comp[i]->type();
    det->component(i)->setGeographicalID(temp);
  }
}




