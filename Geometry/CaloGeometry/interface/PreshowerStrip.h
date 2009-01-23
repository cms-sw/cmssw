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

$Date: 2008/11/10 15:20:15 $
$Revision: 1.6 $
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

      virtual std::vector<HepPoint3D> vocalCorners( const double* pv,
						    HepPoint3D&   ref ) const 
      { return localCorners( pv, ref ) ; }

      static std::vector<HepPoint3D> localCorners( const double* pv, 
						   HepPoint3D&   ref ) ;
      virtual HepTransform3D getTransform( std::vector<HepPoint3D>* lptr ) const
      { return HepTransform3D() ; }

   private:
};

std::ostream& operator<<( std::ostream& s , const PreshowerStrip& cell) ;

#endif
