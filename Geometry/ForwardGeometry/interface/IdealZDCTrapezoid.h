#ifndef Geometry_ForwardGeometry_IdealZDCTrapezoid_H
#define Geometry_ForwardGeometry_IdealZDCTrapezoid_H 1

#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

namespace calogeom {
  /** \class IdealZDCTrapezoid
    
  Trapezoid class used for ZDC volumes.  
  
  Required parameters for an ideal zdc trapezoid:
  
  - dz, dx, dy 
  - locaton x, y and z of faceCenter
  - tilt angle of z faces
  
  Total: 7 parameters 
  
  $Date: 2007/08/28 18:10:10 $
  $Revision: 1.1 $
  \author E. Garcia - UIC
  */
  class IdealZDCTrapezoid: public CaloCellGeometry 
  {
     public:

	IdealZDCTrapezoid( const GlobalPoint& faceCenter,
			   float              an , 
			   float              dx , 
			   float              dy , 
			   float              dz ,
			   const CornersMgr*  mgr           ) :  
	   CaloCellGeometry ( faceCenter, mgr ) ,
	   m_an             ( an ) ,    
	   m_dx             ( dx ) ,
	   m_dy             ( dy ) ,
	   m_dz             ( dz )    {}
    
	virtual ~IdealZDCTrapezoid() {}

	virtual bool inside( const GlobalPoint & point ) const;  

	virtual const CornersVec& getCorners() const;

	const float an() const { return m_an ; }
	const float dx() const { return m_dx ; }
	const float dy() const { return m_dy ; }
	const float dz() const { return m_dz ; }
    
     private:

	float m_an ;
	float m_dx ;
	float m_dy ;
	float m_dz ;
  };

   std::ostream& operator<<( std::ostream& s , const IdealZDCTrapezoid& cell ) ;
}

#endif
