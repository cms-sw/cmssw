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

$Date: 2007/04/12 17:47:43 $
$Revision: 1.3 $
\author F. Cossutti
   
*/


class PreshowerStrip : public CaloCellGeometry
{
   public:

      PreshowerStrip( const GlobalPoint& po ,
		      float              dx , 
		      float              dy , 
		      float              dz ,
		      const CornersMgr*  mgr ) :
	 CaloCellGeometry ( po , mgr ) ,
	 m_dx             ( dx ) ,
	 m_dy             ( dy ) ,
	 m_dz             ( dz ) {}

      virtual ~PreshowerStrip() {}

      virtual bool inside( const GlobalPoint& p ) const ;
  
      virtual const CornersVec& getCorners() const ;

      const float dx() const { return m_dx ; }
      const float dy() const { return m_dy ; }
      const float dz() const { return m_dz ; }

   protected:

   private:

      float m_dx ;
      float m_dy ;
      float m_dz ;
};

std::ostream& operator<<( std::ostream& s , const PreshowerStrip& cell) ;

#endif
