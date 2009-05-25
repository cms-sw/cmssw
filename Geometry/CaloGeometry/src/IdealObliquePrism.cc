#include "Geometry/CaloGeometry/interface/IdealObliquePrism.h"
#include <math.h>

namespace calogeom {
   
   static GlobalPoint etaPhiR( float eta, float phi, float rad )
   {
      return GlobalPoint( rad*cosf(phi)/coshf(eta) , 
			  rad*sinf(phi)/coshf(eta) ,
			  rad*tanhf(eta)            ) ;
   }
   
   static GlobalPoint etaPhiPerp( float eta, float phi, float perp )  
   {
      return GlobalPoint( perp*cosf(phi) ,
			  perp*sinf(phi) ,
			  perp*sinhf(eta) ) ;
   }

   static GlobalPoint etaPhiZ(float eta, float phi, float z) 
   {
      return GlobalPoint( z*cosf(phi)/sinhf(eta) , 
			  z*sinf(phi)/sinhf(eta) ,
			  z                       ) ;
   }


   std::vector<HepGeom::Point3D<double> >
   IdealObliquePrism::localCorners( const double* pv  ,
				    HepGeom::Point3D<double> &   ref   )
   {
      assert( 0 != pv ) ;

      const double dEta ( pv[0] ) ;
      const double dPhi ( pv[1] ) ;
      const double dz   ( pv[2] ) ;
      const double eta  ( pv[3] ) ;
      const double z    ( pv[4] ) ;

      std::vector<GlobalPoint> gc ( 8, GlobalPoint(0,0,0) ) ;
      std::vector<HepGeom::Point3D<double> >  lc ( 8, HepGeom::Point3D<double> ( 0,0,0) ) ;

      const GlobalPoint p ( etaPhiZ( eta, 0, z ) ) ;

      if( 0 < dz )
      {
	 const float r_near ( p.perp()/cos( dPhi ) ) ;
	 const float r_far  ( r_near*( ( p.mag() + 2*dz )/p.mag() ) ) ;
	 gc[ 0 ] = etaPhiPerp( eta + dEta , +dPhi , r_near ) ; // (+,+,near)
	 gc[ 1 ] = etaPhiPerp( eta + dEta , -dPhi , r_near ) ; // (+,-,near)
	 gc[ 2 ] = etaPhiPerp( eta - dEta , -dPhi , r_near ) ; // (-,-,near)
	 gc[ 3 ] = etaPhiPerp( eta - dEta , +dPhi , r_near ) ; // (-,+,far)
	 gc[ 4 ] = etaPhiPerp( eta + dEta , +dPhi , r_far  ) ; // (+,+,far)
	 gc[ 5 ] = etaPhiPerp( eta + dEta , -dPhi , r_far  ) ; // (+,-,far)
	 gc[ 6 ] = etaPhiPerp( eta - dEta , -dPhi , r_far  ) ; // (-,-,far)
	 gc[ 7 ] = etaPhiPerp( eta - dEta , +dPhi , r_far  ) ; // (-,+,far)
      }
      else
      {
	 const float z_near ( z ) ;
	 const float z_far  ( z*( 1 - 2*dz/p.mag() ) ) ;
	 gc[ 0 ] = etaPhiZ( eta + dEta , +dPhi , z_near ) ; // (+,+,near)
	 gc[ 1 ] = etaPhiZ( eta + dEta , -dPhi , z_near ) ; // (+,-,near)
	 gc[ 2 ] = etaPhiZ( eta - dEta , -dPhi , z_near ) ; // (-,-,near)
	 gc[ 3 ] = etaPhiZ( eta - dEta , +dPhi , z_near ) ; // (-,+,far)
	 gc[ 4 ] = etaPhiZ( eta + dEta , +dPhi , z_far  ) ; // (+,+,far)
	 gc[ 5 ] = etaPhiZ( eta + dEta , -dPhi , z_far  ) ; // (+,-,far)
	 gc[ 6 ] = etaPhiZ( eta - dEta , -dPhi , z_far  ) ; // (-,-,far)
	 gc[ 7 ] = etaPhiZ( eta - dEta , +dPhi , z_far  ) ; // (-,+,far)
      }
      for( unsigned int i ( 0 ) ; i != 8 ; ++i )
      {
	 lc[i] = HepGeom::Point3D<double> ( gc[i].x(), gc[i].y(), gc[i].z() ) ;
      }

      ref   = 0.25*( lc[0] + lc[1] + lc[2] + lc[3] ) ;
      return lc ;
   }

   const CaloCellGeometry::CornersVec& 
   IdealObliquePrism::getCorners() const
   {
      const CornersVec& co ( CaloCellGeometry::getCorners() ) ;
      if( co.uninitialized() )
      {
	 CornersVec& corners ( setCorners() ) ;
	 if( dz()>0 ) 
	 { 
	    /* In this case, the faces are parallel to the zaxis.  
	       This implies that all corners will have the same 
	       cylindrical radius. 
	    */
	    const GlobalPoint p      ( getPosition() ) ;
	    const float       r_near ( p.perp()/cos(dPhi()) ) ;
	    const float       r_far  ( r_near*( ( p.mag() + 2*dz() )/p.mag() ) ) ;
	    const float       eta    ( p.eta() ) ;
	    const float       phi    ( p.phi() ) ;
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
	    const float       z_near ( p.z() ) ;
	    const float       mag    ( p.mag() ) ;
	    const float       z_far  ( z_near*( 1 - 2*dz()/mag ) ) ; // negative to correct sign
	    const float       eta    ( p.eta() ) ;
	    const float       phi    ( p.phi() ) ;
	    
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
      return co ;
   }

   bool 
   IdealObliquePrism::inside( const GlobalPoint& point ) const 
   {
      // first check eta/phi
      bool is_inside ( false ) ;

      const GlobalPoint& face ( getPosition() ) ;

      // eta
      if( fabs( point.eta() - face.eta() ) <= dEta()  &&
	  fabs( point.phi() - face.phi() ) <= dPhi()     )
      {
	 // distance (trickier)
	 GlobalPoint face2 ( etaPhiR( face.eta(), 
				      face.phi(), 
				      face.mag() + 2*fabs( dz() ) ) );
	 if( dz() > 0 ) 
	 { // 
	    const float projection ( point.perp()*cos( point.phi() - face.phi() ) ) ;
	    is_inside = ( projection >= face.perp() &&
			  projection <= face2.perp()    ) ;
	 } 
	 else 
	 { // here, it is just a Z test.
	    is_inside = ( ( ( face.z()<0 ) ? ( point.z()<=face.z()  ) :
			    ( point.z()>=face.z()  ) ) && // "front" face
			  ( ( face.z()<0 ) ? ( point.z()>=face2.z() ) :
			    ( point.z()<=face2.z() ) ) ); // "back" face
	 }
      }
      return is_inside;
   }

   std::ostream& operator<<( std::ostream& s, const IdealObliquePrism& cell ) 
   {
      s << "Center: " <<  cell.getPosition() << std::endl ;
      s << "dEta = " << cell.dEta() << ", dPhi = " << cell.dPhi() << ", dz = " << cell.dz() << std::endl ;
      return s;
}

}
