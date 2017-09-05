#include "Geometry/CaloGeometry/interface/PreshowerStrip.h"
#include <iostream>

typedef PreshowerStrip::CCGFloat CCGFloat ;
typedef PreshowerStrip::Pt3D     Pt3D     ;
typedef PreshowerStrip::Pt3DVec  Pt3DVec  ;
typedef PreshowerStrip::Tr3D     Tr3D     ;


PreshowerStrip::PreshowerStrip()
  : CaloCellGeometry()
{}

PreshowerStrip::PreshowerStrip( const PreshowerStrip& tr ) 
  : CaloCellGeometry( tr )
{
  *this = tr ; 
}

PreshowerStrip::~PreshowerStrip() 
{}

PreshowerStrip& 
PreshowerStrip::operator=( const PreshowerStrip& tr ) 
{
  if( &tr != this )
  {
    CaloCellGeometry::operator=( tr ) ;
  }
  return *this ; 
}

void
PreshowerStrip::initCorners(CaloCellGeometry::CornersVec& corners) 
{
  if( corners.uninitialized() ) 
  {
    const GlobalPoint& ctr ( getPosition() ) ;
    const CCGFloat x ( ctr.x() ) ;
    const CCGFloat y ( ctr.y() ) ;
    const CCGFloat z ( ctr.z() ) ;

    const double st ( sin(tilt()) ) ;
    const double ct ( cos(tilt()) ) ;

    for( unsigned int ix ( 0 ) ; ix !=2 ; ++ix )
    {
      const double sx ( 0 == ix ? -1.0 : +1.0 ) ;
      for( unsigned int iy ( 0 ) ; iy !=2 ; ++iy )
      {
	const double sy ( 0 == iy ? -1.0 : +1.0 ) ;
	for( unsigned int iz ( 0 ) ; iz !=2 ; ++iz )
	{
	  const double sz ( 0 == iz ? -1.0 : +1.0 ) ;
	  const unsigned int  i ( 4*iz + 2*ix + 
				  ( 1 == ix ? 1-iy : iy ) ) ;//keeps ordering same as before

	  corners[ i ] = GlobalPoint( 
	    dy()>dx() ? 
	    x + sx*dx() : 
	    x + sx*dx()*ct - sz*dz()*st ,
	    dy()<dx() ? 
	    y + sy*dy() : 
	    y + sy*dy()*ct - sz*dz()*st ,
	    dy()>dx() ? 
	    z + sz*dz()*ct + sy*dy()*st :
	    z + sz*dz()*ct + sx*dx()*st ) ;
	}
      }
    }
  }
}

std::ostream& operator<<( std::ostream& s, const PreshowerStrip& cell ) 
{
  s << "Center: " <<  cell.getPosition() << std::endl ;
  if( cell.param() != nullptr )
  {
    s << "dx = " << cell.dx() << ", dy = " << cell.dy() << ", dz = " << cell.dz() << std::endl ;

    const CaloCellGeometry::CornersVec& corners ( cell.getCorners() ) ; 
    for( unsigned int ci ( 0 ) ; ci != corners.size(); ci++ ) 
    {
      s  << "Corner: " << corners[ci] << std::endl;
    }
  }
  else
  {
    s << " with empty parameters." << std::endl;
  }
  
  return s;
}

void
PreshowerStrip::localCorners( Pt3DVec&        lc  ,
			      const CCGFloat* pv  ,
			      Pt3D&           ref  )
{
  assert( 8 == lc.size() ) ;
  assert( nullptr != pv ) ;

  const CCGFloat dx ( pv[0] ) ;
  const CCGFloat dy ( pv[1] ) ;
  const CCGFloat dz ( pv[2] ) ;

  lc[0] = Pt3D( -dx, -dy, -dz ) ;
  lc[1] = Pt3D( -dx,  dy, -dz ) ;
  lc[2] = Pt3D(  dx,  dy, -dz ) ;
  lc[3] = Pt3D(  dx, -dy, -dz ) ;
  lc[4] = Pt3D( -dx, -dy,  dz ) ;
  lc[5] = Pt3D( -dx,  dy,  dz ) ;
  lc[6] = Pt3D(  dx,  dy,  dz ) ;
  lc[7] = Pt3D(  dx, -dy,  dz ) ;

  ref   = Pt3D(0,0,0) ;
}
