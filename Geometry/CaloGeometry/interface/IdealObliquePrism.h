#ifndef GEOMETRY_CALOGEOMETRY_IDEALOBLIQUEPRISM_H
#define GEOMETRY_CALOGEOMETRY_IDEALOBLIQUEPRISM_H 1

#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

namespace calogeom {
  /** \class IdealObliquePrism
    
  Oblique prism class used for HCAL  (HB, HE, HO) volumes.
  
  Required parameters for an ideal oblique prism:
  
  - eta, phi of axis
  - radial distance (along axis) to front and back faces
  - single bit - faces parallel or perpendicular to z-axis
  - eta width and phi width of faces (same for front/back)
  
  Total: 6+1 parameters
  
  Internally, the "point of reference" is the center (eta/phi) of the
  front face of the prism.  Therefore, the only internally stored
  parameters are eta and phi widths, the axis tower thickness, and the
  parallel/perpendicular setting.  The parallel/perpendicular setting
  is encoded in the sign of the thickness.  (positive = parallel to
  z-axis, negative = perpendicular)

  $Date: 2008/11/10 15:20:15 $
  $Revision: 1.6 $
  \author J. Mans - Minnesota
  */
   class IdealObliquePrism : public CaloCellGeometry 
   {
      public:

	 IdealObliquePrism( const GlobalPoint& faceCenter, 
			    const CornersMgr*  mgr       ,
			    const double*      parm       ) : 
	    CaloCellGeometry ( faceCenter, mgr, parm ) {}

	 virtual ~IdealObliquePrism() {}

	 virtual const CornersVec& getCorners() const ;

	 virtual bool inside( const GlobalPoint & point ) const ;  

	 double dEta()  const { return param()[0] ; }
	 double dPhi()  const { return param()[1] ; }
	 double dz()    const { return param()[2] ; }
	 double eta()   const { return param()[3] ; }
	 double z()     const { return param()[4] ; }

	 static std::vector<HepPoint3D> localCorners( const double* pv,
						      HepPoint3D&   ref ) ;

	 virtual std::vector<HepPoint3D> vocalCorners( const double* pv,
						       HepPoint3D&   ref ) const 
	 { return localCorners( pv, ref ) ; }

      private:
   };

   std::ostream& operator<<( std::ostream& s , const IdealObliquePrism& cell ) ;
}


#endif
