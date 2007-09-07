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

   $Date: 2007/09/05 19:53:09 $
   $Revision: 1.2 $
   \author J. Mans - Minnesota
   */
   class IdealZPrism : public CaloCellGeometry 
   {
      public:

	 IdealZPrism( const GlobalPoint& faceCenter , 
                      const CornersMgr*  mgr ,
		      const float*       parm          )  : 
	    CaloCellGeometry ( faceCenter, mgr ),
	    m_parms          ( parm )                        {}

	 virtual ~IdealZPrism() {}

	 virtual bool inside( const GlobalPoint & point ) const ;

	 virtual const CornersVec& getCorners()           const ;

	 float wEta() const { return param()[0] ; }
	 float wPhi() const { return param()[1] ; }
	 float dz()   const { return param()[2] ; }

      private:

	 const float* param() const { return m_parms ; }

	 const float*      m_parms ;
   };

   std::ostream& operator<<( std::ostream& s , const IdealZPrism& cell ) ;
}


#endif
