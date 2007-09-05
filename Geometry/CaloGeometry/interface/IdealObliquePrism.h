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

  $Date: 2005/10/03 22:35:23 $
  $Revision: 1.2 $
  \author J. Mans - Minnesota
  */
   class IdealObliquePrism : public CaloCellGeometry 
   {
      public:

	 IdealObliquePrism( const GlobalPoint& faceCenter, 
			    float              wEta, 
			    float              wPhi, 
			    float              thick, 
			    bool               parallelToZaxis,
			    const CornersMgr*  mgr               ) : 
	    CaloCellGeometry ( faceCenter, mgr ) ,
	    m_wEta           ( wEta/2 ) ,
	    m_wPhi           ( wPhi/2 ) ,
	    m_thick          ( parallelToZaxis ? thick : -thick ) {}

	 virtual ~IdealObliquePrism() {}

	 virtual const CornersVec& getCorners() const ;

	 virtual bool inside( const GlobalPoint & point ) const ;  

	 float wEta()  const { return m_wEta ; }
	 float wPhi()  const { return m_wPhi ; }
	 float thick() const { return m_thick ; }

      private:
	 
	 float m_wEta  ;
	 float m_wPhi  ; // half-widths
	 float m_thick ;
   };

   std::ostream& operator<<( std::ostream& s , const IdealObliquePrism& cell ) ;
}


#endif
