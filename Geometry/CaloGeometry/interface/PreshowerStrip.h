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

$Date: 2011/09/27 09:10:38 $
$Revision: 1.13 $
\author F. Cossutti
   
*/


class PreshowerStrip : public CaloCellGeometry
{
public:

  typedef CaloCellGeometry::CCGFloat CCGFloat ;
  typedef CaloCellGeometry::Pt3D     Pt3D     ;
  typedef CaloCellGeometry::Pt3DVec  Pt3DVec  ;
  typedef CaloCellGeometry::Tr3D     Tr3D     ;

  PreshowerStrip() ;

  PreshowerStrip( const PreshowerStrip& tr ) ;

  PreshowerStrip& operator=( const PreshowerStrip& tr ) ;

  PreshowerStrip( const GlobalPoint& po   ,
		  const CornersMgr*  mgr  ,
		  const CCGFloat*    parm  ) :
    CaloCellGeometry ( po , mgr, parm ) {}

  virtual ~PreshowerStrip();

  virtual const CornersVec& getCorners() const ;

  CCGFloat dx() const { return param()[0] ; }
  CCGFloat dy() const { return param()[1] ; }
  CCGFloat dz() const { return param()[2] ; }
  CCGFloat tilt() const { return param()[3] ; }

  virtual void vocalCorners( Pt3DVec&        vec ,
			     const CCGFloat* pv  ,
			     Pt3D&           ref  ) const 
    { localCorners( vec, pv, ref ) ; }

  static void localCorners( Pt3DVec&        vec ,
			    const CCGFloat* pv  , 
			    Pt3D&           ref  ) ;

  virtual Tr3D getTransform( Pt3DVec* /*lptr*/ ) const
    { return Tr3D() ; }
};

std::ostream& operator<<( std::ostream& s , const PreshowerStrip& cell) ;

#endif
