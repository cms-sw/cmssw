#ifndef PreshowerStrip_h
#define PreshowerStrip_h

#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include <CLHEP/Geometry/Point3D.h>
#include <CLHEP/Geometry/Plane3D.h>
#include <CLHEP/Geometry/Vector3D.h>
#include <CLHEP/Geometry/Transform3D.h>
#include <vector>


/**

   \class PreshowerStrip

   \brief A base class to handle the shape of preshower strips.

$Date: 2010/04/20 17:23:11 $
$Revision: 1.9 $
\author F. Cossutti
   
*/


class PreshowerStrip : public CaloCellGeometry
{
   public:

      PreshowerStrip( const GlobalPoint& po ,
		      const CornersMgr*  mgr,
		      const double*      parm ) :
	 CaloCellGeometry ( po , mgr, parm ) {}

      virtual ~PreshowerStrip() {}

      virtual const CornersVec& getCorners() const ;

      const double dx() const { return param()[0] ; }
      const double dy() const { return param()[1] ; }
      const double dz() const { return param()[2] ; }
      const double tilt() const { return param()[3] ; }

      virtual std::vector<HepGeom::Point3D<double> > vocalCorners( const double* pv,
						    HepGeom::Point3D<double> &   ref ) const 
      { return localCorners( pv, ref ) ; }

      static std::vector<HepGeom::Point3D<double> > localCorners( const double* pv, 
						   HepGeom::Point3D<double> &   ref ) ;
      virtual HepGeom::Transform3D getTransform( std::vector<HepGeom::Point3D<double> >* lptr ) const
      { return HepGeom::Transform3D() ; }

   private:
};

std::ostream& operator<<( std::ostream& s , const PreshowerStrip& cell) ;

#endif
