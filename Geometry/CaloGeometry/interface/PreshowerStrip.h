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

$Date: 2007/09/07 19:08:34 $
$Revision: 1.5 $
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

      virtual bool inside( const GlobalPoint& p ) const ;
  
      virtual const CornersVec& getCorners() const ;

      const double dx() const { return param()[0] ; }
      const double dy() const { return param()[1] ; }
      const double dz() const { return param()[2] ; }

      static std::vector<HepPoint3D> localCorners( const double* pv, 
						   HepPoint3D&   ref ) ;

   private:
};

std::ostream& operator<<( std::ostream& s , const PreshowerStrip& cell) ;

#endif
