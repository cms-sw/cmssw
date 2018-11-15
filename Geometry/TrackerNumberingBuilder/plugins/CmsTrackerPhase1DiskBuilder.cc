#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerPhase1DiskBuilder.h"
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


bool
CmsTrackerPhase1DiskBuilder::PhiSort( const GeometricDet* Panel1, const GeometricDet* Panel2 )
{
  return( Panel1->phi() < Panel2->phi());
}

void
CmsTrackerPhase1DiskBuilder::PhiPosNegSplit_innerOuter( std::vector< GeometricDet const *>::iterator begin,
							std::vector< GeometricDet const *>::iterator end )
{
  // first sort in phi, lowest first (-pi to +pi)
  std::sort( begin, end, PhiSort );

  // now put positive phi (in order) ahead of negative phi as in std geometry
  std::vector<const GeometricDet*> theCompsPosNeg;
  theCompsPosNeg.empty();
  theCompsPosNeg.clear();
  // also find the average radius (used to split inner and outer disk panels)
  double theRmin = (**begin).rho();
  double theRmax = theRmin;
  for(vector<const GeometricDet*>::const_iterator it=begin;
      it!=end;it++){
    if((**it).phi() >= 0) theCompsPosNeg.emplace_back(*it);
    theRmin = std::min( theRmin, (**it).rho());
    theRmax = std::max( theRmax, (**it).rho());
  }
  for(vector<const GeometricDet*>::const_iterator it=begin;
      it!=end;it++){
    if((**it).phi() < 0) theCompsPosNeg.emplace_back(*it);
  }

  // now put inner disk panels first
  //  double radius_split = 0.5 * (theRmin + theRmax);
  // force the split radius to be 100 mm to be able to deal with disks with only outer ring
  double radius_split = 100.;
  std::vector<const GeometricDet*> theCompsInnerOuter;
  theCompsInnerOuter.empty();
  theCompsInnerOuter.clear();
  unsigned int num_inner = 0;
  for(vector<const GeometricDet*>::const_iterator it=theCompsPosNeg.begin();
      it!=theCompsPosNeg.end();it++){
    if((**it).rho() <= radius_split) {
      theCompsInnerOuter.emplace_back(*it);
      num_inner++;
    }
  }

  for(vector<const GeometricDet*>::const_iterator it=theCompsPosNeg.begin();
      it!=theCompsPosNeg.end();it++){
    if((**it).rho() > radius_split) theCompsInnerOuter.emplace_back(*it);
  }
  //  std::cout << "num of inner = " << num_inner << " with radius less than " << radius_split << std::endl;
  // now shift outer by one

  std::rotate(theCompsInnerOuter.begin()+num_inner,theCompsInnerOuter.end()-1,theCompsInnerOuter.end());
  std::rotate(theCompsInnerOuter.begin(),theCompsInnerOuter.begin()+num_inner-1,theCompsInnerOuter.begin()+num_inner);
  std::copy(theCompsInnerOuter.begin(), theCompsInnerOuter.end(), begin);
}

void
CmsTrackerPhase1DiskBuilder::buildComponent( DDFilteredView& fv, GeometricDet* g, std::string s )
{
  CmsTrackerPanelBuilder theCmsTrackerPanelBuilder;
  GeometricDet * subdet = new GeometricDet( &fv, theCmsTrackerStringToEnum.type( ExtractStringFromDDD::getString( s, &fv )));

  switch( theCmsTrackerStringToEnum.type( ExtractStringFromDDD::getString( s, &fv )))
  {
  case GeometricDet::panel:
    theCmsTrackerPanelBuilder.build( fv, subdet, s );
    break;
  default:
    edm::LogError( "CmsTrackerPhase1DiskBuilder" ) << " ERROR - I was expecting a Panel, I got a " << ExtractStringFromDDD::getString( s, &fv );   
  }  
  g->addComponent( subdet );
}

void
CmsTrackerPhase1DiskBuilder::sortNS( DDFilteredView& fv, GeometricDet* det )
{


  GeometricDet::ConstGeometricDetContainer & comp = det->components();

  switch( det->components().front()->type())
  {
  case GeometricDet::panel:
    PhiPosNegSplit_innerOuter( comp.begin(), comp.end());
    break;
  default:
    edm::LogError( "CmsTrackerPhase1DiskBuilder" ) << "ERROR - wrong SubDet to sort..... " << det->components().front()->type();
  }

  GeometricDet::GeometricDetContainer zminpanels;  // Here z refers abs(z);
  GeometricDet::GeometricDetContainer zmaxpanels;  // So, zmin panel is always closer to ip.

  uint32_t totalblade = comp.size()/2;
  //  std::cout << "pixel_disk " << pixel_disk << endl; 

  zminpanels.reserve( totalblade );
  zmaxpanels.reserve( totalblade );
  for( uint32_t j = 0; j < totalblade; j++ )
  {
    if( std::abs( comp[2*j]->translation().z()) > std::abs( comp[ 2*j +1 ]->translation().z()))
    {
      zmaxpanels.emplace_back( det->component(2*j) );
      zminpanels.emplace_back( det->component(2*j+1) );

    }
    else if( std::abs( comp[2*j]->translation().z()) < std::abs( comp[ 2*j +1 ]->translation().z()))
    {
      zmaxpanels.emplace_back( det->component(2*j+1) );
      zminpanels.emplace_back( det->component(2*j) );
    }
    else
    {
      edm::LogWarning( "CmsTrackerPhase1DiskBuilder" ) << "WARNING - The Z of  both panels are equal! ";
    }
  }

  for( uint32_t fn = 0; fn < zminpanels.size(); fn++ )
  {
    uint32_t blade = fn + 1;
    uint32_t panel = 2; // though being zmin, it is actually the one facing away the ip
    uint32_t temp = ( blade << 2 ) | panel;
    zminpanels[fn]->setGeographicalID( temp );
  }
  
  for( uint32_t bn = 0; bn < zmaxpanels.size(); bn++)
  {
    uint32_t blade = bn + 1;
    uint32_t panel = 1; // though being zmax, it is the one facing the ip
    uint32_t temp = ( blade << 2) | panel;
    zmaxpanels[bn]->setGeographicalID( temp );
  }
  
  det->clearComponents();
  det->addComponents( zminpanels );
  det->addComponents( zmaxpanels );

}

