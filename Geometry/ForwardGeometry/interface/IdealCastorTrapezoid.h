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
  
  $Date: 2008/11/10 15:27:02 $
  $Revision: 1.2 $
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

	const double dx() const { return param()[0] ; }
	const double dy() const { return param()[1] ; }
	const double dz() const { return param()[2] ; }
    
     private:
  };

   std::ostream& operator<<( std::ostream& s , const IdealCastorTrapezoid& cell ) ;
}

#endif
