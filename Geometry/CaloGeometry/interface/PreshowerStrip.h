#ifndef GEOMETRY_CALOGEOMETRY_PRESHOWERSTRIP_H
#define GEOMETRY_CALOGEOMETRY_PRESHOWERSTRIP_H

#include "DataFormats/CaloGeometry/interface/CaloCellGeometry.h"

/**

   \class PreshowerStrip

   \brief A base class to handle the shape of preshower strips.

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
		        CornersMgr*  mgr  ,
		  const CCGFloat*    parm  ) :
    CaloCellGeometry ( po , mgr, parm ) {initSpan();}

  virtual ~PreshowerStrip();

  CCGFloat dx() const { return param()[0] ; }
  CCGFloat dy() const { return param()[1] ; }
  CCGFloat dz() const { return param()[2] ; }
  CCGFloat tilt() const { return param()[3] ; }

  virtual void vocalCorners( Pt3DVec&        vec ,
			     const CCGFloat* pv  ,
			     Pt3D&           ref  ) const override 
    { localCorners( vec, pv, ref ) ; }

  static void localCorners( Pt3DVec&        vec ,
			    const CCGFloat* pv  , 
			    Pt3D&           ref  ) ;

  virtual void getTransform( Tr3D& tr, Pt3DVec* /*lptr*/ ) const override
    { }

 private:
  virtual void initCorners(CaloCellGeometry::CornersVec&) override;
};

std::ostream& operator<<( std::ostream& s , const PreshowerStrip& cell) ;

#endif
