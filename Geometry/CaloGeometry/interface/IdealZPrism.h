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

   $Date: 2009/01/23 15:03:24 $
   $Revision: 1.6 $
   \author J. Mans - Minnesota
   */
   class IdealZPrism : public CaloCellGeometry 
   {
      public:

	 IdealZPrism( const GlobalPoint& faceCenter , 
                      const CornersMgr*  mgr ,
		      const double*      parm          )  : 
	    CaloCellGeometry ( faceCenter, mgr, parm )   {}

	 virtual ~IdealZPrism() {}

	 virtual bool inside( const GlobalPoint & point ) const ;

	 virtual const CornersVec& getCorners()           const ;

	 double dEta() const { return param()[0] ; }
	 double dPhi() const { return param()[1] ; }
	 double dz()   const { return param()[2] ; }
	 double eta()  const { return param()[3] ; }
	 double z()    const { return param()[4] ; }

	 static std::vector<HepGeom::Point3D<double> > localCorners( const double* pv,
						      HepGeom::Point3D<double> &   ref ) ;

	 virtual std::vector<HepGeom::Point3D<double> > vocalCorners( const double* pv,
						       HepGeom::Point3D<double> &   ref ) const 
	 { return localCorners( pv, ref ) ; }

      private:
   };

   std::ostream& operator<<( std::ostream& s , const IdealZPrism& cell ) ;
}


#endif
