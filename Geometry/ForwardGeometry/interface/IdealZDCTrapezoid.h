#ifndef Geometry_ForwardGeometry_IdealZDCTrapezoid_H
#define Geometry_ForwardGeometry_IdealZDCTrapezoid_H 1

#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

  /** \class IdealZDCTrapezoid
    
  Trapezoid class used for ZDC volumes.  
  
  Required parameters for an ideal zdc trapezoid:
  
  - dz, dx, dy 
  - locaton x, y and z of faceCenter
  - tilt angle of z faces
  
  Total: 7 parameters 
  
  $Revision: 1.9 $
  \author E. Garcia - UIC
  */

class IdealZDCTrapezoid: public CaloCellGeometry 
{
   public:

      typedef CaloCellGeometry::CCGFloat CCGFloat ;
      typedef CaloCellGeometry::Pt3D     Pt3D     ;
      typedef CaloCellGeometry::Pt3DVec  Pt3DVec  ;
      typedef CaloCellGeometry::Tr3D     Tr3D     ;

      IdealZDCTrapezoid() ;

      IdealZDCTrapezoid( const IdealZDCTrapezoid& idzt ) ;
      
      IdealZDCTrapezoid& operator=( const IdealZDCTrapezoid& idzt ) ;

      IdealZDCTrapezoid( const GlobalPoint& faceCenter,
			       CornersMgr*  mgr       ,
			 const CCGFloat*    parm        ) ;
	 
      virtual ~IdealZDCTrapezoid() ;

      CCGFloat an() const ;
      CCGFloat dx() const ;
      CCGFloat dy() const ;
      CCGFloat dz() const ;
      CCGFloat ta() const ;
      CCGFloat dt() const ;

      virtual void vocalCorners( Pt3DVec&        vec ,
				 const CCGFloat* pv  ,
				 Pt3D&           ref  ) const ;

      static void localCorners( Pt3DVec&        vec ,
				const CCGFloat* pv  , 
				Pt3D&           ref  ) ;
    
   private:
      void initCorners(CaloCellGeometry::CornersVec& );


};

std::ostream& operator<<( std::ostream& s , const IdealZDCTrapezoid& cell ) ;

#endif
