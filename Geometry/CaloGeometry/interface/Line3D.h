// Hep-LIKE geometrical 3D LINE class
//
//
// Author: BKH
//

#ifndef HepLine3D_hh
#define HepLine3D_hh

#include "CLHEP/Geometry/Point3D.h"
#include "CLHEP/Geometry/Normal3D.h"
#include "CLHEP/Geometry/Plane3D.h"
#include <iostream>

typedef double  HepDouble;
typedef bool    HepBoolean;
typedef HepGeom::Point3D<double> HepPoint3D;
typedef HepGeom::Plane3D<double> HepPlane3D;
typedef HepGeom::Vector3D<double> HepVector3D;

class HepLine3D 
{
   protected:
      HepPoint3D  pp ;
      HepVector3D uu ;
      HepDouble   eps ;

   public:

      HepLine3D( const HepPoint3D&  p, 
		 const HepVector3D& v,
		 HepDouble          sml = 1.e-10 ) :
	 pp  ( p ), 
	 uu  ( v*( v.mag()>1.e-10 ? 1./v.mag() : 1 ) ), 
	 eps ( fabs( sml ) ) {} 

      HepLine3D( const HepPoint3D& p1, 
		 const HepPoint3D& p2, 
		 HepDouble         sml = 1.e-10 ) :
	 pp  ( p1 ), 
	 uu  ( (p2-p1)*( (p2-p1).mag()>1.e-10 ? 1./(p2-p1).mag() : 1 ) ),
	 eps ( fabs( sml ) ) {} 

      // Copy constructor
      HepLine3D( const HepLine3D& line ) :
	 pp (line.pp), uu(line.uu), eps(line.eps) {}
      
      // Destructor
      ~HepLine3D() {};
      
      // Assignment
      HepLine3D& operator=(const HepLine3D& line) 
      {
	 pp = line.pp; uu = line.uu; eps = line.eps; return *this;
      }

      // Test for equality
      HepBoolean operator == (const HepLine3D& l) const 
      {
	 return pp == l.pp && uu == l.uu ;
      }

      // Test for inequality
      HepBoolean operator != (const HepLine3D& l) const 
      {
	 return pp != l.pp || uu != l.uu ;
      }

      const HepPoint3D&  pt() const { return pp ; }

      const HepVector3D& uv() const { return uu ; }

      HepPoint3D point( const HepGeom::Plane3D<double>& pl, HepBoolean& parallel ) const
      {
	 const HepDouble num ( -pl.d() - pl.a()*pp.x() - pl.b()*pp.y() - pl.c()*pp.z() ) ;
	 const HepDouble den (           pl.a()*uu.x() + pl.b()*uu.y() + pl.c()*uu.z() ) ;

	 parallel = ( eps > fabs( num ) ) || ( eps > fabs( den ) ) ;

	 return ( parallel ? pp : HepPoint3D( pp + uu*(num/den) ) ) ;
      }

      HepPoint3D point( const HepPoint3D& q ) const
      {
	 return ( pp + ( ( q.x() - pp.x() )*uu.x() +
			 ( q.y() - pp.y() )*uu.y() +
			 ( q.z() - pp.z() )*uu.z() )*uu ) ; 
      }

      HepDouble dist2( const HepPoint3D& q ) const { return ( q - point( q ) ).mag2() ; }
      HepDouble dist( const HepPoint3D& q ) const { return ( q - point( q ) ).mag() ; }
};

#endif
