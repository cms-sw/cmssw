#include "Geometry/ForwardGeometry/interface/IdealZDCTrapezoid.h"
#include <math.h>

typedef CaloCellGeometry::CCGFloat CCGFloat ;
typedef CaloCellGeometry::Pt3D     Pt3D     ;
typedef CaloCellGeometry::Pt3DVec  Pt3DVec  ;

IdealZDCTrapezoid::IdealZDCTrapezoid()
  : CaloCellGeometry()
{
}

IdealZDCTrapezoid::IdealZDCTrapezoid( const IdealZDCTrapezoid& idzt ) 
  : CaloCellGeometry( idzt )
{
   *this = idzt ;
}

IdealZDCTrapezoid& 
IdealZDCTrapezoid::operator=( const IdealZDCTrapezoid& idzt ) 
{
   if( &idzt != this ) CaloCellGeometry::operator=( idzt ) ;
   return *this ;
}

IdealZDCTrapezoid::IdealZDCTrapezoid( const GlobalPoint& faceCenter,
				            CornersMgr*  mgr       ,
				      const CCGFloat*    parm        ) :  
   CaloCellGeometry ( faceCenter, mgr, parm )  
{initSpan();}
	 
IdealZDCTrapezoid::~IdealZDCTrapezoid() {}

CCGFloat 
IdealZDCTrapezoid::an() const { return param()[0] ; }

CCGFloat 
IdealZDCTrapezoid::dx() const 
{
   return param()[1] ; 
}

CCGFloat 
IdealZDCTrapezoid::dy() const 
{
   return param()[2] ; 
}

CCGFloat 
IdealZDCTrapezoid::dz() const 
{
   return param()[3] ; 
}

CCGFloat 
IdealZDCTrapezoid::ta() const 
{
   return tan( an() ) ; 
}

CCGFloat 
IdealZDCTrapezoid::dt() const 
{
   return dy()*ta() ; 
}

void 
IdealZDCTrapezoid::vocalCorners( Pt3DVec&        vec ,
				 const CCGFloat* pv  ,
				 Pt3D&           ref  ) const 
{
   localCorners( vec, pv, ref ) ; 
}

void
IdealZDCTrapezoid::localCorners( Pt3DVec&        lc  ,
				 const CCGFloat* pv  ,
				 Pt3D&           ref   )
{
   assert( 8 == lc.size() ) ;
   assert( 0 != pv ) ;

   const CCGFloat an ( pv[0] ) ;
   const CCGFloat dx ( pv[1] ) ;
   const CCGFloat dy ( pv[2] ) ;
   const CCGFloat dz ( pv[3] ) ;
   const CCGFloat ta ( tan( an ) ) ;
   const CCGFloat dt ( dy*ta ) ;

   lc[0] = Pt3D ( -dx, -dy, +dz+dt ) ;
   lc[1] = Pt3D ( -dx, +dy, +dz-dt ) ;
   lc[2] = Pt3D ( +dx, +dy, +dz-dt ) ;
   lc[3] = Pt3D ( +dx, -dy, +dz+dt ) ;
   lc[4] = Pt3D ( -dx, -dy, -dz+dt ) ;
   lc[5] = Pt3D ( -dx, +dy, -dz-dt ) ;
   lc[6] = Pt3D ( +dx, +dy, -dz-dt ) ;
   lc[7] = Pt3D ( +dx, -dy, -dz+dt ) ;

   ref   = 0.25*( lc[0] + lc[1] + lc[2] + lc[3] ) ;
}

void
IdealZDCTrapezoid::initCorners(CaloCellGeometry::CornersVec& corners)
{
   if( corners.uninitialized() ) 
   {
      const GlobalPoint& p ( getPosition() ) ;
      const CCGFloat zsign ( 0 < p.z() ? 1. : -1. ) ;
      const Pt3D  gf ( p.x(), p.y(), p.z() ) ;

      Pt3D  lf ;
      Pt3DVec lc ( 8, Pt3D(0,0,0) ) ;
      localCorners( lc, param(), lf ) ;
      const Pt3D  lb ( lf.x() , lf.y() , lf.z() - 2.*dz() ) ;
      const Pt3D  ls ( lf.x() - dx(), lf.y(), lf.z() ) ;
      
      const Pt3D  gb ( gf.x() , gf.y() , gf.z() + 2.*zsign*dz() ) ;

      const Pt3D  gs ( gf.x() - zsign*dx(),
		       gf.y() ,
		       gf.z()         ) ;

      const HepGeom::Transform3D tr ( lf, lb, ls,
				      gf, gb, gs ) ;

      for( unsigned int i ( 0 ) ; i != 8 ; ++i )
      {
	 const Pt3D  gl ( tr*lc[i] ) ;
	 corners[i] = GlobalPoint( gl.x(), gl.y(), gl.z() ) ;
      }
   }
}

std::ostream& operator<<( std::ostream& s, const IdealZDCTrapezoid& cell ) 
{
   s << "Center: " <<  cell.getPosition() << std::endl ;
   s << "TiltAngle = " << cell.an()*180./M_PI << " deg, dx = " 
     << cell.dx() 
     << ", dy = " << cell.dy() << ", dz = " << cell.dz() << std::endl ;
   return s;
}
