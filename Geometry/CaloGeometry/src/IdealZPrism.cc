#include "Geometry/CaloGeometry/interface/IdealZPrism.h"
#include <math.h>

namespace calogeom 
{
   static GlobalPoint etaPhiR( float eta ,
			       float phi ,
			       float rad   ) 
   {
      return GlobalPoint( rad*cosf(  phi )/coshf( eta ) ,
			  rad*sinf(  phi )/coshf( eta ) ,
			  rad*tanhf( eta )             ) ;
   }

   static GlobalPoint etaPhiPerp( float eta , 
				  float phi , 
				  float perp  ) 
   {
      return GlobalPoint( perp*cosf(  phi ) , 
			  perp*sinf(  phi ) , 
			  perp*sinhf( eta ) );
   }

   static GlobalPoint etaPhiZ( float eta , 
			       float phi ,
			       float z    ) 
   {
      return GlobalPoint( z*cosf( phi )/sinhf( eta ) ,
			  z*sinf( phi )/sinhf( eta ) ,
			  z                            ) ;
   }

   const CaloCellGeometry::CornersVec& 
   IdealZPrism::getCorners() const 
   {
      const CornersVec& co ( CaloCellGeometry::getCorners() ) ;
      if( co.empty() ) 
      {
	 CornersVec& corners ( setCorners() ) ;

	 const GlobalPoint p      ( getPosition() ) ;
	 const float       z_near ( p.z() ) ;
	 const float       z_far  ( z_near + m_dz*p.z()/fabs( p.z() ) ) ;
	 const float       eta    ( p.eta() ) ;
	 const float       phi    ( p.phi() ) ;

	 corners[ 0 ] = etaPhiZ( eta + m_wEta, phi + m_wPhi, z_near ); // (+,+,near)
	 corners[ 1 ] = etaPhiZ( eta + m_wEta, phi - m_wPhi, z_near ); // (+,-,near)
	 corners[ 2 ] = etaPhiZ( eta - m_wEta, phi - m_wPhi, z_near ); // (-,-,near)
	 corners[ 3 ] = etaPhiZ( eta - m_wEta, phi + m_wPhi, z_near ); // (-,+,near)
	 corners[ 4 ] = GlobalPoint( corners[0].x(), corners[0].y(), z_far ); // (+,+,far)
	 corners[ 5 ] = GlobalPoint( corners[1].x(), corners[1].y(), z_far ); // (+,-,far)
	 corners[ 6 ] = GlobalPoint( corners[2].x(), corners[2].y(), z_far ); // (-,-,far)
	 corners[ 7 ] = GlobalPoint( corners[3].x(), corners[3].y(), z_far ); // (-,+,far)	
      }
      return co ;
   }

   bool 
   IdealZPrism::inside( const GlobalPoint & point ) const 
   {
      return ( fabs( point.eta() - getPosition().eta() ) <= m_wEta &&
	       fabs( point.phi() - getPosition().phi() ) <= m_wPhi &&
	       fabs( point.z()   - getPosition().z()   ) <= m_dz       ) ; 
   }

   std::ostream& operator<<( std::ostream& s, const IdealZPrism& cell ) 
   {
      s << "Center: " <<  cell.getPosition() << std::endl ;
      s << "wEta = " << cell.wEta() << ", wPhi = " << cell.wPhi() << ", dz = " << cell.dz() << std::endl ;
      return s;
   }

}
