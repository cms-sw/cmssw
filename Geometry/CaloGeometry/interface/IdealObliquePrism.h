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

  $Date: 2007/09/05 19:53:09 $
  $Revision: 1.3 $
  \author J. Mans - Minnesota
  */
   class IdealObliquePrism : public CaloCellGeometry 
   {
      public:

	 IdealObliquePrism( const GlobalPoint& faceCenter, 
			    const CornersMgr*  mgr       ,
			    const float*       parm       ) : 
	    CaloCellGeometry ( faceCenter, mgr ) ,
	    m_parms          ( parm )               {}

	 virtual ~IdealObliquePrism() {}

	 virtual const CornersVec& getCorners() const ;

	 virtual bool inside( const GlobalPoint & point ) const ;  

	 float wEta()  const { return param()[0] ; }
	 float wPhi()  const { return param()[1] ; }
	 float thick() const { return param()[2] ; }

      private:

	 const float* param() const { return m_parms ; }

	 const   float*      m_parms ;
   };

   std::ostream& operator<<( std::ostream& s , const IdealObliquePrism& cell ) ;
}


#endif
