#ifndef GEOMETRY_CALOGEOMETRY_IDEALZPRISM_H
#define GEOMETRY_CALOGEOMETRY_IDEALZPRISM_H 1

#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

namespace calogeom 
{
   /** \class IdealZPrism
    
   Prism class used for HF volumes.  HF volumes are prisms with axes along the Z direction whose
   face shapes are set by 
  
   Required parameters for an ideal Z prism:
  
   - eta, phi of axis
   - Z location of front and back faces
   - eta width and phi width of frontface
  
   Total: 6 parameters
  
   Internally, the "point of reference" is the center (eta/phi) of the
   front face of the prism.  Therefore, the only internally stored
   parameters are eta and phi HALF-widths and the tower z thickness.

   $Date: 2005/10/28 18:08:35 $
   $Revision: 1.1 $
   \author J. Mans - Minnesota
   */
   class IdealZPrism : public CaloCellGeometry 
   {
      public:

	 IdealZPrism( const GlobalPoint& faceCenter , 
		      float              wEta, 
		      float              wPhi, 
		      float              dz  ,
                      const CornersMgr*  mgr)  : 
	    CaloCellGeometry ( faceCenter, mgr ) ,
	    m_wEta           ( wEta/2 ) ,
	    m_wPhi           ( wPhi/2 ) ,
	    m_dz             ( dz     )   {}

	 virtual ~IdealZPrism() {}

	 virtual bool inside( const GlobalPoint & point ) const ;

	 virtual const CornersVec& getCorners()           const ;

	 float wEta() const { return m_wEta ; }
	 float wPhi() const { return m_wPhi ; }
	 float dz()   const { return m_dz   ; }

      private:

	 float m_wEta ;
	 float m_wPhi ; // half-widths
	 float m_dz ;
   };

   std::ostream& operator<<( std::ostream& s , const IdealZPrism& cell ) ;
}


#endif
