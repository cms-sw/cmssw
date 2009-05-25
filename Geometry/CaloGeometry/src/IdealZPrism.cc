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

   std::vector<HepGeom::Point3D<double> >
   IdealZPrism::localCorners( const double* pv  ,
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

      const float z_near ( z ) ;
      const float z_far  ( z*( 1 - 2*dz/p.mag() ) ) ;
      gc[ 0 ] = etaPhiZ( eta + dEta , +dPhi , z_near ) ; // (+,+,near)
      gc[ 1 ] = etaPhiZ( eta + dEta , -dPhi , z_near ) ; // (+,-,near)
      gc[ 2 ] = etaPhiZ( eta - dEta , -dPhi , z_near ) ; // (-,-,near)
      gc[ 3 ] = etaPhiZ( eta - dEta , +dPhi , z_near ) ; // (-,+,far)
      gc[ 4 ] = GlobalPoint( gc[0].x(), gc[0].y(), z_far ); // (+,+,far)
      gc[ 5 ] = GlobalPoint( gc[1].x(), gc[1].y(), z_far ); // (+,-,far)
      gc[ 6 ] = GlobalPoint( gc[2].x(), gc[2].y(), z_far ); // (-,-,far)
      gc[ 7 ] = GlobalPoint( gc[3].x(), gc[3].y(), z_far ); // (-,+,far)	

      for( unsigned int i ( 0 ) ; i != 8 ; ++i )
      {
	 lc[i] = HepGeom::Point3D<double> ( gc[i].x(), gc[i].y(), gc[i].z() ) ;
      }

      ref   = 0.25*( lc[0] + lc[1] + lc[2] + lc[3] ) ;
      return lc ;
   }

   const CaloCellGeometry::CornersVec& 
   IdealZPrism::getCorners() const 
   {
      const CornersVec& co ( CaloCellGeometry::getCorners() ) ;
      if( co.uninitialized() ) 
      {
	 CornersVec& corners ( setCorners() ) ;

	 const GlobalPoint p      ( getPosition() ) ;
	 const float       z_near ( p.z() ) ;
	 const float       z_far  ( z_near + 2*dz()*p.z()/fabs( p.z() ) ) ;
	 const float       eta    ( p.eta() ) ;
	 const float       phi    ( p.phi() ) ;

	 corners[ 0 ] = etaPhiZ( eta + dEta(), phi + dPhi(), z_near ); // (+,+,near)
	 corners[ 1 ] = etaPhiZ( eta + dEta(), phi - dPhi(), z_near ); // (+,-,near)
	 corners[ 2 ] = etaPhiZ( eta - dEta(), phi - dPhi(), z_near ); // (-,-,near)
	 corners[ 3 ] = etaPhiZ( eta - dEta(), phi + dPhi(), z_near ); // (-,+,near)
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
      return ( fabs( point.eta() - getPosition().eta() ) <= dEta() &&
	       fabs( point.phi() - getPosition().phi() ) <= dPhi() &&
	       fabs( point.z()   - getPosition().z()   ) <= dz()       ) ; 
   }

   std::ostream& operator<<( std::ostream& s, const IdealZPrism& cell ) 
   {
      s << "Center: " <<  cell.getPosition() << std::endl ;
      s << "dEta = " << cell.dEta() << ", dPhi = " << cell.dPhi() << ", dz = " << cell.dz() << std::endl ;
      return s;
   }

}
