#include "DetectorDescription/Core/src/ExtrudedPolygon.h" 
#include "DetectorDescription/Core/interface/DDUnits.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/src/Solid.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <cassert>

using DDI::ExtrudedPolygon;
using namespace dd::operators;

ExtrudedPolygon::ExtrudedPolygon( const std::vector<double> & x,
				  const std::vector<double> & y,
				  const std::vector<double> & z,
				  const std::vector<double> & zx,
				  const std::vector<double> & zy,
				  const std::vector<double> & zscale)
  : Solid( DDSolidShape::ddextrudedpolygon )	      
{
  assert( x.size() == y.size());
  assert( z.size() == zx.size());
  assert( z.size() == zy.size());
  assert( z.size() == zscale.size());

  p_.reserve( x.size() + y.size() + z.size() + zx.size() + zy.size() + zscale.size() + 1 );
  p_.emplace_back( z.size());
  p_.insert( p_.end(), x.begin(), x.end());
  p_.insert( p_.end(), y.begin(), y.end());
  p_.insert( p_.end(), z.begin(), z.end());
  p_.insert( p_.end(), zx.begin(), zx.end());
  p_.insert( p_.end(), zy.begin(), zy.end());
  p_.insert( p_.end(), zscale.begin(), zscale.end());
}	      

double ExtrudedPolygon::volume() const
{
  double volume = 0;
  /* FIXME: volume is not implemented */ 
   return volume;
}

void
DDI::ExtrudedPolygon::stream( std::ostream & os ) const
{
  auto xysize = ( unsigned int )(( p_.size() - 4*p_[0]) * 0.5 );
  os << " XY Points[cm]=";
  for( unsigned int k = 1; k <= xysize; ++k )
    os << CONVERT_TO( p_[k], cm ) << ", " <<  CONVERT_TO( p_[k + xysize], cm ) << "; ";
  os << " with " << p_[0] << " Z sections:";
  unsigned int m0 = p_.size() - 4*p_[0];
  for( unsigned int m = m0; m < m0 + p_[0]; ++m )
  {
    os << " z[cm]=" << CONVERT_TO( p_[m], cm );
    os << ", x[cm]=" << CONVERT_TO( p_[m+p_[0]], cm );
    os << ", y[cm]="  << CONVERT_TO( p_[m+2*p_[0]], cm );
    os << ", scale[cm]="  << CONVERT_TO( p_[m+3*p_[0]], cm ) << ";";
  }
}
