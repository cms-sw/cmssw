#include "Geometry/CaloGeometry/interface/IdealObliquePrism.h"
#include <cmath>

typedef IdealObliquePrism::CCGFloat CCGFloat ;
typedef IdealObliquePrism::Pt3D     Pt3D     ;
typedef IdealObliquePrism::Pt3DVec  Pt3DVec  ;

IdealObliquePrism::IdealObliquePrism() :
   CaloCellGeometry()
{}

IdealObliquePrism::IdealObliquePrism( const IdealObliquePrism& idop ) 
  : CaloCellGeometry( idop )
{
  *this = idop ;
}

IdealObliquePrism& 
IdealObliquePrism::operator=( const IdealObliquePrism& idop ) 
{
  if( &idop != this ) CaloCellGeometry::operator=( idop ) ;
  return *this ;
}

IdealObliquePrism::IdealObliquePrism( const GlobalPoint& faceCenter, 
				      CornersMgr*        mgr       ,
				      const CCGFloat*    parm       )
  : CaloCellGeometry ( faceCenter, mgr, parm )
{initSpan();}

IdealObliquePrism::~IdealObliquePrism() 
{}

CCGFloat 
IdealObliquePrism::dEta()  const 
{
   return param()[IdealObliquePrism::k_dEta] ;
}

CCGFloat 
IdealObliquePrism::dPhi()  const 
{ 
   return param()[IdealObliquePrism::k_dPhi] ;
}

CCGFloat 
IdealObliquePrism::dz()    const 
{
   return param()[IdealObliquePrism::k_dZ] ;
}

CCGFloat 
IdealObliquePrism::eta()   const
{
   return param()[IdealObliquePrism::k_Eta] ; 
}

CCGFloat 
IdealObliquePrism::z()     const 
{
   return param()[IdealObliquePrism::k_Z] ;
}

void 
IdealObliquePrism::vocalCorners( Pt3DVec&        vec ,
				 const CCGFloat* pv  ,
				 Pt3D&           ref  ) const 
{ 
   localCorners( vec, pv, ref ) ; 
}

GlobalPoint
IdealObliquePrism::etaPhiPerp( float eta, float phi, float perp )  
{
   return GlobalPoint( perp*cosf(phi) ,
		       perp*sinf(phi) ,
		       perp*sinhf(eta) ) ;
}

GlobalPoint 
IdealObliquePrism::etaPhiZ(float eta, float phi, float z) 
{
   return GlobalPoint( z*cosf(phi)/sinhf(eta) , 
		       z*sinf(phi)/sinhf(eta) ,
		       z                       ) ;
}

void IdealObliquePrism::localCorners( Pt3DVec&        lc  ,
				      const CCGFloat* pv  ,
				      Pt3D&           ref   )
{
   assert( 8 == lc.size() ) ;
   assert( nullptr != pv ) ;
   
   const CCGFloat dEta ( pv[IdealObliquePrism::k_dEta] ) ;
   const CCGFloat dPhi ( pv[IdealObliquePrism::k_dPhi] ) ;
   const CCGFloat dz   ( pv[IdealObliquePrism::k_dZ] ) ;
   const CCGFloat eta  ( pv[IdealObliquePrism::k_Eta] ) ;
   const CCGFloat z    ( pv[IdealObliquePrism::k_Z] ) ;
   
   std::vector<GlobalPoint> gc ( 8, GlobalPoint(0,0,0) ) ;

   const GlobalPoint p ( etaPhiZ( eta, 0, z ) ) ;

   if( 0 < dz )
   {
      const CCGFloat r_near ( p.perp()/cos( dPhi ) ) ;
      const CCGFloat r_far  ( r_near*( ( p.mag() + 2*dz )/p.mag() ) ) ;
      gc[ 0 ] = etaPhiPerp( eta + dEta , +dPhi , r_near ) ; // (+,+,near)
      gc[ 1 ] = etaPhiPerp( eta + dEta , -dPhi , r_near ) ; // (+,-,near)
      gc[ 2 ] = etaPhiPerp( eta - dEta , -dPhi , r_near ) ; // (-,-,near)
      gc[ 3 ] = etaPhiPerp( eta - dEta , +dPhi , r_near ) ; // (-,+,near)
      gc[ 4 ] = etaPhiPerp( eta + dEta , +dPhi , r_far  ) ; // (+,+,far)
      gc[ 5 ] = etaPhiPerp( eta + dEta , -dPhi , r_far  ) ; // (+,-,far)
      gc[ 6 ] = etaPhiPerp( eta - dEta , -dPhi , r_far  ) ; // (-,-,far)
      gc[ 7 ] = etaPhiPerp( eta - dEta , +dPhi , r_far  ) ; // (-,+,far)
   }
   else
   {
      const CCGFloat z_near ( z ) ;
      const CCGFloat z_far  ( z*( 1 - 2*dz/p.mag() ) ) ;
      gc[ 0 ] = etaPhiZ( eta + dEta , +dPhi , z_near ) ; // (+,+,near)
      gc[ 1 ] = etaPhiZ( eta + dEta , -dPhi , z_near ) ; // (+,-,near)
      gc[ 2 ] = etaPhiZ( eta - dEta , -dPhi , z_near ) ; // (-,-,near)
      gc[ 3 ] = etaPhiZ( eta - dEta , +dPhi , z_near ) ; // (-,+,near)
      gc[ 4 ] = etaPhiZ( eta + dEta , +dPhi , z_far  ) ; // (+,+,far)
      gc[ 5 ] = etaPhiZ( eta + dEta , -dPhi , z_far  ) ; // (+,-,far)
      gc[ 6 ] = etaPhiZ( eta - dEta , -dPhi , z_far  ) ; // (-,-,far)
      gc[ 7 ] = etaPhiZ( eta - dEta , +dPhi , z_far  ) ; // (-,+,far)
   }
   for( unsigned int i ( 0 ) ; i != 8 ; ++i )
   {
      lc[i] = Pt3D( gc[i].x(), gc[i].y(), gc[i].z() ) ;
   }
   
   ref   = 0.25*( lc[0] + lc[1] + lc[2] + lc[3] ) ;
}

void IdealObliquePrism::initCorners(CaloCellGeometry::CornersVec& co)
{
   if( co.uninitialized() )
   {
      CornersVec& corners ( co ) ;
      if( dz()>0 ) 
      { 
	 /* In this case, the faces are parallel to the zaxis.  
	    This implies that all corners will have the same 
	    cylindrical radius. 
	 */
	 const GlobalPoint p      ( getPosition() ) ;
	 const CCGFloat    r_near ( p.perp()/cos(dPhi()) ) ;
	 const CCGFloat    r_far  ( r_near*( ( p.mag() + 2*dz() )/p.mag() ) ) ;
	 const CCGFloat    eta    ( p.eta() ) ;
	 const CCGFloat    phi    ( p.phi() ) ;
	 corners[ 0 ] = etaPhiPerp( eta + dEta() , phi + dPhi() , r_near ) ; // (+,+,near)
	 corners[ 1 ] = etaPhiPerp( eta + dEta() , phi - dPhi() , r_near ) ; // (+,-,near)
	 corners[ 2 ] = etaPhiPerp( eta - dEta() , phi - dPhi() , r_near ) ; // (-,-,near)
	 corners[ 3 ] = etaPhiPerp( eta - dEta() , phi + dPhi() , r_near ) ; // (-,+,near)
	 corners[ 4 ] = etaPhiPerp( eta + dEta() , phi + dPhi() , r_far ) ; // (+,+,far)
	 corners[ 5 ] = etaPhiPerp( eta + dEta() , phi - dPhi() , r_far ) ; // (+,-,far)
	 corners[ 6 ] = etaPhiPerp( eta - dEta() , phi - dPhi() , r_far ) ; // (-,-,far)
	 corners[ 7 ] = etaPhiPerp( eta - dEta() , phi + dPhi() , r_far ) ; // (-,+,far)
      } 
      else 
      {
	 /* In this case, the faces are perpendicular to the zaxis.  
	    This implies that all corners will have the same 
	    z-dimension. 
	 */
	 const GlobalPoint p      ( getPosition() ) ;
	 const CCGFloat    z_near ( p.z() ) ;
	 const CCGFloat    mag    ( p.mag() ) ;
	 const CCGFloat    z_far  ( z_near*( 1 - 2*dz()/mag ) ) ; // negative to correct sign
	 const CCGFloat    eta    ( p.eta() ) ;
	 const CCGFloat    phi    ( p.phi() ) ;
	 
	 corners[ 0 ] = etaPhiZ( eta + dEta(), phi + dPhi(), z_near ) ; // (+,+,near)
	 corners[ 1 ] = etaPhiZ( eta + dEta(), phi - dPhi(), z_near ) ; // (+,-,near)
	 corners[ 2 ] = etaPhiZ( eta - dEta(), phi - dPhi(), z_near ) ; // (-,-,near)
	 corners[ 3 ] = etaPhiZ( eta - dEta(), phi + dPhi(), z_near ) ; // (-,+,near)
	 corners[ 4 ] = etaPhiZ( eta + dEta(), phi + dPhi(), z_far  ) ; // (+,+,far)
	 corners[ 5 ] = etaPhiZ( eta + dEta(), phi - dPhi(), z_far  ) ; // (+,-,far)
	 corners[ 6 ] = etaPhiZ( eta - dEta(), phi - dPhi(), z_far  ) ; // (-,-,far)
	 corners[ 7 ] = etaPhiZ( eta - dEta(), phi + dPhi(), z_far  ) ; // (-,+,far)
	 
      }
   }
}

std::ostream& operator<<( std::ostream& s, const IdealObliquePrism& cell ) 
{
   s << "Center: " <<  cell.getPosition() << std::endl ;
   s << "dEta = " << cell.dEta() << ", dPhi = " << cell.dPhi() << ", dz = " << cell.dz() << std::endl ;
   return s;
}
