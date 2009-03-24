#ifndef Geometry_ForwardGeometry_IdealCastorTrapezoid_H
#define Geometry_ForwardGeometry_IdealCastorTrapezoid_H 1

#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

namespace calogeom {
  /** \class IdealCastorTrapezoid
    
  Trapezoid class used for CASTOR volumes.  
  
  Required parameters for an ideal trapezoid:
  
  - dxl, dxh, dh, dz, z-face tilt-angle, dR
    dxl is the HALFlength of the side at smaller y
    dxh is the HALFlength of the side at larger y
    dxl and dxh are both either positive or negative;
        positive means a "right-handed" trapezoid cross section
        negative means a "left-handed" trapezoid cross section
    dh is the HALFheight of the actual side, not its projection
    dz is the HALFlength in z of each long side
    an is the angle of tilt in the z direction 
    dR is the length of the line in the xy plane from the origin
        to the perpendicular intersection with the extended edge
        at the lower-y side of the trapezoid

  - locaton x, y and z of faceCenter
  
  Total: 7 parameters 
  
  $Date: 2009/01/24 03:59:46 $
  $Revision: 1.5 $
  \author P. Katsas - UoA
  */
   class IdealCastorTrapezoid: public CaloCellGeometry 
   {
      public:

	 IdealCastorTrapezoid( const GlobalPoint& faceCenter,
			       const CornersMgr*  mgr       ,
			       const double*      parm        ) :  
	    CaloCellGeometry ( faceCenter, mgr, parm )  {}
	 
	 virtual ~IdealCastorTrapezoid() {}
	 
	 virtual bool inside( const GlobalPoint & point ) const;  

	 virtual const CornersVec& getCorners() const;

	 const double dxl() const { return param()[0] ; }
	 const double dxh() const { return param()[1] ; }
	 const double dx()  const { return ( dxl()+dxh() )/2. ; }
	 const double dh()  const { return param()[2] ; }
	 const double dy()  const { return dh()*sin(an()) ; }
	 const double dz()  const { return param()[3] ; }
	 const double dhs() const { return dh()*cos(an()) ; }
	 const double dzb() const { return dz() + dhs() ; }
	 const double dzs() const { return dz() - dhs() ; }
	 const double an()  const { return param()[4] ; }
	 const double dR()  const { return param()[5] ; }

	 virtual std::vector<HepPoint3D> vocalCorners( const double* pv,
						       HepPoint3D&   ref ) const 
	 { return localCorners( pv, ref ) ; }

	 static std::vector<HepPoint3D> localCorners( const double* pv, 
						      HepPoint3D&   ref ) ;
	 virtual HepTransform3D getTransform( std::vector<HepPoint3D>* lptr ) const
	 { return HepTransform3D() ; }
     private:
  };

   std::ostream& operator<<( std::ostream& s , const IdealCastorTrapezoid& cell ) ;
}

#endif
