#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerDiskBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerPanelBuilder.h"
#include "Geometry/TrackerNumberingBuilder/plugins/TrackerStablePhiSort.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>
#include <algorithm>

using namespace std;

CmsTrackerDiskBuilder::CmsTrackerDiskBuilder( unsigned int totalBlade )
  : m_totalBlade( totalBlade )
{}

bool
PhiSort( const GeometricDet* Panel1, const GeometricDet* Panel2 )
{
  return( Panel1->phi() < Panel2->phi());
}

void
CmsTrackerDiskBuilder::PhiPosNegSplit_innerOuter( GeometricDet::ConstGeometricDetContainer::iterator begin,
						  GeometricDet::ConstGeometricDetContainer::iterator end )
{
  // first sort in phi, lowest first (-pi to +pi)
  std::sort( begin, end, PhiSort );

  // now put positive phi (in order) ahead of negative phi as in std geometry
  GeometricDet::ConstGeometricDetContainer theCompsPosNeg;
  theCompsPosNeg.empty();
  theCompsPosNeg.clear();
  // also find the average radius (used to split inner and outer disk panels)
  double theRmin = (**begin).rho();
  double theRmax = theRmin;
  for(GeometricDet::ConstGeometricDetContainer::const_iterator it=begin;
      it!=end;it++){
    if((**it).phi() >= 0) theCompsPosNeg.push_back(*it);
    theRmin = std::min( theRmin, (**it).rho());
    theRmax = std::max( theRmax, (**it).rho());
  }
  for(GeometricDet::ConstGeometricDetContainer::const_iterator it=begin;
      it!=end;it++){
    if((**it).phi() < 0) theCompsPosNeg.push_back(*it);
  }

  // now put inner disk panels first
  double radius_split = 0.5 * (theRmin + theRmax);
  GeometricDet::ConstGeometricDetContainer theCompsInnerOuter;
  theCompsInnerOuter.empty();
  theCompsInnerOuter.clear();
  //unsigned int num_inner = 0;
  for(GeometricDet::ConstGeometricDetContainer::const_iterator it=theCompsPosNeg.begin();
      it!=theCompsPosNeg.end();it++){
    if((**it).rho() <= radius_split) {
      theCompsInnerOuter.push_back(*it);
      //num_inner++;
    }
  }
  for(GeometricDet::ConstGeometricDetContainer::const_iterator it=theCompsPosNeg.begin();
      it!=theCompsPosNeg.end();it++){
    if((**it).rho() > radius_split) theCompsInnerOuter.push_back(*it);
  }
  //std::cout << "num of inner = " << num_inner << " with radius less than " << radius_split << std::endl;
  std::copy(theCompsInnerOuter.begin(), theCompsInnerOuter.end(), begin);
}

void
CmsTrackerDiskBuilder::buildComponent( DDFilteredView& fv, GeometricDet* g, std::string s )
{
  CmsTrackerPanelBuilder theCmsTrackerPanelBuilder;
  GeometricDet * subdet = new GeometricDet( &fv, theCmsTrackerStringToEnum.type( ExtractStringFromDDD::getString( s, &fv )));

  switch( theCmsTrackerStringToEnum.type( ExtractStringFromDDD::getString( s, &fv )))
  {
  case GeometricDet::panel:
    theCmsTrackerPanelBuilder.build( fv, subdet, s );
    break;
  default:
    edm::LogError( "CmsTrackerDiskBuilder" ) << " ERROR - I was expecting a Panel, I got a " << ExtractStringFromDDD::getString( s, &fv );   
  }  
  g->addComponent( subdet );
}

void
CmsTrackerDiskBuilder::sortNS( DDFilteredView& fv, GeometricDet* det )
{
  GeometricDet::ConstGeometricDetContainer & comp = det->components();

  switch( det->components().front()->type())
  {
  case GeometricDet::panel:
    if( m_totalBlade == 24 )
      TrackerStablePhiSort( comp.begin(), comp.end(), ExtractPhi());
    else
      PhiPosNegSplit_innerOuter( comp.begin(), comp.end());
    break;
  default:
    edm::LogError( "CmsTrackerDiskBuilder" ) << "ERROR - wrong SubDet to sort..... " << det->components().front()->type();
  }

  GeometricDet::GeometricDetContainer zminpanels;  // Here z refers abs(z);
  GeometricDet::GeometricDetContainer zmaxpanels;  // So, zmin panel is always closer to ip.

  uint32_t totalblade = comp.size()/2;
  if( totalblade != m_totalBlade )
    edm::LogError( "CmsTrackerDiskBuilder" ) << "ERROR, The Total Number of Blade in one disk is " << totalblade << "; configured " << m_totalBlade;

  zminpanels.reserve( totalblade );
  zmaxpanels.reserve( totalblade );
  for( uint32_t j = 0; j < totalblade; j++ )
  {
    if( fabs( comp[2*j]->translation().z()) > fabs( comp[ 2*j +1 ]->translation().z()))
    {
      zmaxpanels.push_back( det->component(2*j) );
      zminpanels.push_back( det->component(2*j+1) );

    }
    else if( fabs( comp[2*j]->translation().z()) < fabs( comp[ 2*j +1 ]->translation().z()))
    {
      zmaxpanels.push_back( det->component(2*j+1) );
      zminpanels.push_back( det->component(2*j) );
    }
    else
    {
      edm::LogWarning( "CmsTrackerDiskBuilder" ) << "WARNING - The Z of  both panels are equal! ";
    }
  }

  for( uint32_t fn = 0; fn < zminpanels.size(); fn++ )
  {
    uint32_t blade = fn + 1;
    uint32_t panel = 1;
    uint32_t temp = ( blade << 2 ) | panel;
    zminpanels[fn]->setGeographicalID( temp );
  }
  
  for( uint32_t bn = 0; bn < zmaxpanels.size(); bn++)
  {
    uint32_t blade = bn + 1;
    uint32_t panel = 2;
    uint32_t temp = ( blade << 2) | panel;
    zmaxpanels[bn]->setGeographicalID( temp );
  }
  
  det->clearComponents();
  det->addComponents( zminpanels );
  det->addComponents( zmaxpanels );
}
