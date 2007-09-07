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

$Date: 2007/09/05 19:53:09 $
$Revision: 1.4 $
\author F. Cossutti
   
*/


class PreshowerStrip : public CaloCellGeometry
{
   public:

      PreshowerStrip( const GlobalPoint& po ,
		      const CornersMgr*  mgr,
		      const float*       parm ) :
	 CaloCellGeometry ( po , mgr ) ,
	 m_parms          ( parm ) {}

      virtual ~PreshowerStrip() {}

      virtual bool inside( const GlobalPoint& p ) const ;
  
      virtual const CornersVec& getCorners() const ;

      const float dx() const { return param()[0] ; }
      const float dy() const { return param()[1] ; }
      const float dz() const { return param()[2] ; }

   private:

      const float* param() const { return m_parms ; }

      const float* m_parms ;
};

std::ostream& operator<<( std::ostream& s , const PreshowerStrip& cell) ;

#endif
