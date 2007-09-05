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

   const CaloCellGeometry::CornersVec& 
   IdealObliquePrism::getCorners() const
   {
      const CornersVec& co ( CaloCellGeometry::getCorners() ) ;
      if( co.empty() )
      {
	 CornersVec& corners ( setCorners() ) ;
	 corners.resize( k_cornerSize ) ;

	 if( m_thick>0 ) 
	 { 
	    /* In this case, the faces are parallel to the zaxis.  
	       This implies that all corners will have the same 
	       cylindrical radius. 
	    */
	    const GlobalPoint p      ( getPosition() ) ;
	    const float       r_near ( p.perp()/cos(m_wPhi) ) ;
	    const float       r_far  ( r_near*( ( p.mag() + m_thick )/p.mag() ) ) ;
	    const float       eta    ( p.eta() ) ;
	    const float       phi    ( p.phi() ) ;
	    corners[ 0 ] = etaPhiPerp( eta + m_wEta , phi + m_wPhi , r_near ) ; // (+,+,near)
	    corners[ 1 ] = etaPhiPerp( eta + m_wEta , phi - m_wPhi , r_near ) ; // (+,-,near)
	    corners[ 2 ] = etaPhiPerp( eta - m_wEta , phi - m_wPhi , r_near ) ; // (-,-,near)
	    corners[ 3 ] = etaPhiPerp( eta - m_wEta , phi + m_wPhi , r_near ) ; // (-,+,near)
	    corners[ 4 ] = etaPhiPerp( eta + m_wEta , phi + m_wPhi , r_far ) ; // (+,+,far)
	    corners[ 5 ] = etaPhiPerp( eta + m_wEta , phi - m_wPhi , r_far ) ; // (+,-,far)
	    corners[ 6 ] = etaPhiPerp( eta - m_wEta , phi - m_wPhi , r_far ) ; // (-,-,far)
	    corners[ 7 ] = etaPhiPerp( eta - m_wEta , phi + m_wPhi , r_far ) ; // (-,+,far)
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
	    const float       z_far  ( z_near*( 1 - m_thick/mag ) ) ; // negative to correct sign
	    const float       eta    ( p.eta() ) ;
	    const float       phi    ( p.phi() ) ;
	    
	    corners[ 0 ] = etaPhiZ( eta + m_wEta, phi + m_wPhi, z_near ) ; // (+,+,near)
	    corners[ 1 ] = etaPhiZ( eta + m_wEta, phi - m_wPhi, z_near ) ; // (+,-,near)
	    corners[ 2 ] = etaPhiZ( eta - m_wEta, phi - m_wPhi, z_near ) ; // (-,-,near)
	    corners[ 3 ] = etaPhiZ( eta - m_wEta, phi + m_wPhi, z_near ) ; // (-,+,near)
	    corners[ 4 ] = etaPhiZ( eta + m_wEta, phi + m_wPhi, z_far  ) ; // (+,+,far)
	    corners[ 5 ] = etaPhiZ( eta + m_wEta, phi - m_wPhi, z_far  ) ; // (+,-,far)
	    corners[ 6 ] = etaPhiZ( eta - m_wEta, phi - m_wPhi, z_far  ) ; // (-,-,far)
	    corners[ 7 ] = etaPhiZ( eta - m_wEta, phi + m_wPhi, z_far  ) ; // (-,+,far)
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
      if( fabs( point.eta() - face.eta() ) <= m_wEta  &&
	  fabs( point.phi() - face.phi() ) <= m_wPhi     )
      {
	 // distance (trickier)
	 GlobalPoint face2 ( etaPhiR( face.eta(), 
				      face.phi(), 
				      face.mag() + fabs( m_thick ) ) );
	 if( m_thick > 0 ) 
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
      s << "wEta = " << cell.wEta() << ", wPhi = " << cell.wPhi() << ", thick = " << cell.thick() << std::endl ;
      return s;
}

}
