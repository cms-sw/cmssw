#ifndef Geometry_ForwardGeometry_IdealCastorTrapezoid_H
#define Geometry_ForwardGeometry_IdealCastorTrapezoid_H 1

#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

namespace calogeom {
  /** \class IdealCastorTrapezoid
    
  Trapezoid class used for CASTOR volumes.  
  
  Required parameters for an ideal trapezoid:
  
  - dz, dx, dy 
  - locaton x, y and z of faceCenter
  - tilt angle of z faces
  
  Total: 7 parameters 
  
  $Date: 2007/09/07 19:11:19 $
  $Revision: 1.3 $
  \author P. Katsas - UoA
  */
   class IdealCastorTrapezoid: public CaloCellGeometry 
   {
      public:

	 IdealCastorTrapezoid( const GlobalPoint& faceCenter,
			    const CornersMgr*  mgr       ,
			    const float*       parm        ) :  
	    CaloCellGeometry ( faceCenter, mgr ) ,
	    m_parms          ( parm )                   {}
	 
	virtual ~IdealCastorTrapezoid() {}

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

   std::ostream& operator<<( std::ostream& s , const IdealCastorTrapezoid& cell ) ;
}

#endif
