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
  
  $Date: 2009/04/01 13:55:16 $
  $Revision: 1.6 $
  \author E. Garcia - UIC
  */
   class IdealZDCTrapezoid: public CaloCellGeometry 
   {
      public:

	 IdealZDCTrapezoid( const GlobalPoint& faceCenter,
			    const CornersMgr*  mgr       ,
			    const double*      parm        ) :  
	    CaloCellGeometry ( faceCenter, mgr, parm )  {}
	 
	virtual ~IdealZDCTrapezoid() {}

	virtual bool inside( const GlobalPoint & point ) const;  

	virtual const CornersVec& getCorners() const;

	const double an() const { return param()[0] ; }
	const double dx() const { return param()[1] ; }
	const double dy() const { return param()[2] ; }
	const double dz() const { return param()[3] ; }
	const double ta() const { return tan( an() ) ; }
	const double dt() const { return dy()*ta() ; }

	 virtual std::vector<HepGeom::Point3D<double> > vocalCorners( const double* pv,
						       HepGeom::Point3D<double> &   ref ) const 
	 { return localCorners( pv, ref ) ; }

	 static std::vector<HepGeom::Point3D<double> > localCorners( const double* pv, 
						      HepGeom::Point3D<double> &   ref ) ;
    
     private:
  };

   std::ostream& operator<<( std::ostream& s , const IdealZDCTrapezoid& cell ) ;
}

#endif
