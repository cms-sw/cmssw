#include "Geometry/ForwardGeometry/interface/IdealCastorTrapezoid.h"
#include "CLHEP/Geometry/Plane3D.h"
#include "CLHEP/Geometry/Transform3D.h"
#include <cmath>

typedef CaloCellGeometry::CCGFloat CCGFloat ;
typedef CaloCellGeometry::Pt3D     Pt3D     ;
typedef CaloCellGeometry::Pt3DVec  Pt3DVec  ;
typedef CaloCellGeometry::Tr3D     Tr3D     ;

IdealCastorTrapezoid::IdealCastorTrapezoid()
  : CaloCellGeometry()
{
}

IdealCastorTrapezoid::IdealCastorTrapezoid( const IdealCastorTrapezoid& idct ) 
  : CaloCellGeometry( idct )
{
   *this = idct ;
}

IdealCastorTrapezoid& 
IdealCastorTrapezoid::operator=( const IdealCastorTrapezoid& idct ) 
{
   if( &idct != this ) CaloCellGeometry::operator=( idct ) ;
   return *this ;
}

IdealCastorTrapezoid::IdealCastorTrapezoid( const GlobalPoint& faceCenter,
					          CornersMgr*  mgr       ,
					    const CCGFloat*    parm       )
  : CaloCellGeometry ( faceCenter, mgr, parm )  
{initSpan();}
	 
IdealCastorTrapezoid::~IdealCastorTrapezoid() 
{
}

CCGFloat 
IdealCastorTrapezoid::dxl() const 
{
   return param()[0] ; 
}

CCGFloat 
IdealCastorTrapezoid::dxh() const 
{
   return param()[1] ; 
}

CCGFloat 
IdealCastorTrapezoid::dx()  const 
{
   return ( dxl()+dxh() )/2. ; 
}

CCGFloat 
IdealCastorTrapezoid::dh()  const 
{
   return param()[2] ; 
}

CCGFloat 
IdealCastorTrapezoid::dy()  const 
{
   return dh()*sin(an()) ; 
}

CCGFloat 
IdealCastorTrapezoid::dz()  const 
{
   return param()[3] ; 
}

CCGFloat 
IdealCastorTrapezoid::dhz() const 
{
   return dh()*cos(an()) ; 
}

CCGFloat 
IdealCastorTrapezoid::dzb() const 
{
   return dz() + dhz() ;
}

CCGFloat 
IdealCastorTrapezoid::dzs() const 
{
   return dz() - dhz() ; 
}

CCGFloat 
IdealCastorTrapezoid::an()  const 
{
   return param()[4] ; 
}

CCGFloat 
IdealCastorTrapezoid::dR()  const 
{
   return param()[5] ; 
}

void 
IdealCastorTrapezoid::vocalCorners( Pt3DVec&        vec ,
				 const CCGFloat* pv  ,
				 Pt3D&           ref  ) const 
{ 
   localCorners( vec, pv, ref ) ; 
}

void
IdealCastorTrapezoid::localCorners( Pt3DVec&        lc  ,
				    const CCGFloat* pv  ,
				    Pt3D&           ref   )
{
   assert( 8 == lc.size() ) ;
   assert( nullptr != pv ) ;
   
   const CCGFloat dxl ( pv[0] ) ;
   const CCGFloat dxh ( pv[1] ) ;
   const CCGFloat dh  ( pv[2] ) ;
   const CCGFloat dz  ( pv[3] ) ;
   const CCGFloat an  ( pv[4] ) ;
   const CCGFloat dx  ( ( dxl +dxh )/2. ) ;
   const CCGFloat dy  ( dh*sin(an) ) ;
   const CCGFloat dhz ( dh*cos(an) ) ;
   const CCGFloat dzb ( dz + dhz ) ;
   const CCGFloat dzs ( dz - dhz ) ;

   assert( 0 < (dxl*dxh) ) ;

   lc[ 0 ] = Pt3D (        -dx, -dy ,  dzb ) ;
   lc[ 1 ] = Pt3D (        -dx, +dy ,  dzs ) ;
   lc[ 2 ] = Pt3D ( +2*dxh -dx, +dy ,  dzs ) ;
   lc[ 3 ] = Pt3D ( +2*dxl -dx, -dy ,  dzb ) ;
   lc[ 4 ] = Pt3D (        -dx, -dy , -dzs ) ;
   lc[ 5 ] = Pt3D (        -dx, +dy , -dzb ) ;
   lc[ 6 ] = Pt3D ( +2*dxh -dx, +dy , -dzb ) ;
   lc[ 7 ] = Pt3D ( +2*dxl -dx, -dy , -dzs ) ;

   ref   = 0.25*( lc[0] + lc[1] + lc[2] + lc[3] ) ;
}

void
IdealCastorTrapezoid::initCorners(CaloCellGeometry::CornersVec& corners)
{
   if( corners.uninitialized() ) 
   {
      const GlobalPoint& p ( getPosition() ) ;
      const CCGFloat zsign ( 0 < p.z() ? 1. : -1. ) ;
      const Pt3D  gf ( p.x(), p.y(), p.z() ) ;

      Pt3D  lf ;
      Pt3DVec lc ( 8, Pt3D(0,0,0) ) ;
      localCorners( lc, param(), lf ) ;
      const Pt3D lb ( lf.x() , lf.y() , lf.z() - 2.*dz() ) ;
      const Pt3D ls ( lf.x() - dx(), lf.y(), lf.z() ) ;


      const CCGFloat fphi ( atan( dx()/( dR() + dy() ) ) ) ;
      const Pt3D     gb ( gf.x() , gf.y() , gf.z() + 2.*zsign*dz() ) ;

      const CCGFloat rho ( dR() + dy() ) ;
      const CCGFloat phi ( gf.phi() + fphi ) ;
      const Pt3D gs ( rho*cos(phi) ,
		      rho*sin(phi) ,
		      gf.z()         ) ;

      const Tr3D tr ( lf, lb, ls,
		      gf, gb, gs ) ;

      for( unsigned int i ( 0 ) ; i != 8 ; ++i )
      {
	 const Pt3D gl ( tr*lc[i] ) ;
	 corners[i] = GlobalPoint( gl.x(), gl.y(), gl.z() ) ;
      }
   }
}

std::ostream& operator<<( std::ostream& s, const IdealCastorTrapezoid& cell ) 
{
   s << "Center: " <<  cell.getPosition() << std::endl ;
//      s 	 << ", dx = " << cell.dx() 
//<< "TiltAngle = " << cell.an() 
//	<< ", dy = " << cell.dy() << ", dz = " << cell.dz() << std::endl ;
   return s;
}
