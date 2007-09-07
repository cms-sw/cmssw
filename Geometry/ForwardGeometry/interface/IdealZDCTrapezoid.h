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
  
  $Date: 2007/09/05 20:04:13 $
  $Revision: 1.2 $
  \author E. Garcia - UIC
  */
   class IdealZDCTrapezoid: public CaloCellGeometry 
   {
      public:

	 IdealZDCTrapezoid( const GlobalPoint& faceCenter,
			    const CornersMgr*  mgr       ,
			    const float*       parm        ) :  
	    CaloCellGeometry ( faceCenter, mgr ) ,
	    m_parms          ( parm )                   {}
	 
	virtual ~IdealZDCTrapezoid() {}

	virtual bool inside( const GlobalPoint & point ) const;  

	virtual const CornersVec& getCorners() const;

	const float an() const { return param()[0] ; }
	const float dx() const { return param()[1] ; }
	const float dy() const { return param()[2] ; }
	const float dz() const { return param()[3] ; }
    
     private:

	const float* param() const { return m_parms ; }

	const float* m_parms ;
  };

   std::ostream& operator<<( std::ostream& s , const IdealZDCTrapezoid& cell ) ;
}

#endif
