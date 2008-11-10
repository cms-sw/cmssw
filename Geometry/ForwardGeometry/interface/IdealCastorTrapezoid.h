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
  
  $Date: 2007/10/08 20:54:40 $
  $Revision: 1.1 $
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

	const double an() const { return param()[0] ; }
	const double dx() const { return param()[1] ; }
	const double dy() const { return param()[2] ; }
	const double dz() const { return param()[3] ; }
    
     private:
  };

   std::ostream& operator<<( std::ostream& s , const IdealCastorTrapezoid& cell ) ;
}

#endif
