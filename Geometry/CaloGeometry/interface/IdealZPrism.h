#ifndef GEOMETRY_CALOGEOMETRY_IDEALZPRISM_H
#define GEOMETRY_CALOGEOMETRY_IDEALZPRISM_H 1

#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

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

\author J. Mans - Minnesota
*/
class IdealZPrism : public CaloCellGeometry 
{
   public:
      
      typedef CaloCellGeometry::CCGFloat CCGFloat ;
      typedef CaloCellGeometry::Pt3D     Pt3D     ;
      typedef CaloCellGeometry::Pt3DVec  Pt3DVec  ;
      
      IdealZPrism() ;
      
      IdealZPrism( const IdealZPrism& idzp ) ;
      
      IdealZPrism& operator=( const IdealZPrism& idzp ) ;
      
      IdealZPrism( const GlobalPoint& faceCenter , 
		   CornersMgr*        mgr        ,
		   const CCGFloat*    parm         ) ;
      
      virtual ~IdealZPrism() ;
      
      CCGFloat dEta() const ;
      CCGFloat dPhi() const ;
      CCGFloat dz()   const ;
      CCGFloat eta()  const ;
      CCGFloat z()    const ;
      
      static void localCorners( Pt3DVec&        vec ,
				const CCGFloat* pv  ,
				Pt3D&           ref   ) ;
      
      virtual void vocalCorners( Pt3DVec&        vec ,
				 const CCGFloat* pv  ,
				 Pt3D&           ref   ) const ;
      
   private:

      virtual void initCorners(CornersVec& ) override;
      
      static GlobalPoint etaPhiR( float eta ,
				  float phi ,
				  float rad   ) ;

      static GlobalPoint etaPhiPerp( float eta , 
				     float phi , 
				     float perp  ) ;

      static GlobalPoint etaPhiZ( float eta , 
				  float phi ,
				  float z    ) ;
};

std::ostream& operator<<( std::ostream& s , const IdealZPrism& cell ) ;

#endif
