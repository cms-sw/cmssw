#ifndef Geometry_ForwardGeometry_IdealCastorTrapezoid_H
#define Geometry_ForwardGeometry_IdealCastorTrapezoid_H 1

#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

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
  
  Total: 6+3 parameters 
  
  $Date: 2010/04/20 17:25:13 $
  $Revision: 1.11 $
  \author P. Katsas - UoA
  */
class IdealCastorTrapezoid: public CaloCellGeometry 
{
   public:

      typedef CaloCellGeometry::CCGFloat CCGFloat ;
      typedef CaloCellGeometry::Pt3D     Pt3D     ;
      typedef CaloCellGeometry::Pt3DVec  Pt3DVec  ;
      typedef CaloCellGeometry::Tr3D     Tr3D     ;

      IdealCastorTrapezoid() ;

      IdealCastorTrapezoid( const IdealCastorTrapezoid& idct ) ;
      
      IdealCastorTrapezoid& operator=( const IdealCastorTrapezoid& idct ) ;
      
      IdealCastorTrapezoid( const GlobalPoint& faceCenter,
			    const CornersMgr*  mgr       ,
			    const CCGFloat*      parm        ) ;
	 
      virtual ~IdealCastorTrapezoid() ;
	 
      virtual const CornersVec& getCorners() const;

      const CCGFloat dxl() const ; 
      const CCGFloat dxh() const ; 
      const CCGFloat dx()  const ; 
      const CCGFloat dh()  const ;
      const CCGFloat dy()  const ; 
      const CCGFloat dz()  const ;
      const CCGFloat dhz() const ; 
      const CCGFloat dzb() const ; 
      const CCGFloat dzs() const ;
      const CCGFloat an()  const ;
      const CCGFloat dR()  const ;

      virtual void vocalCorners( Pt3DVec&        vec ,
				 const CCGFloat* pv  ,
				 Pt3D&           ref  ) const ;

      static void localCorners( Pt3DVec&        vec ,
				const CCGFloat* pv  , 
				Pt3D&           ref   ) ;
   private:
};

std::ostream& operator<<( std::ostream& s , const IdealCastorTrapezoid& cell ) ;

#endif
