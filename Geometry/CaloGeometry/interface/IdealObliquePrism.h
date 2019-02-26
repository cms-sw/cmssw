#ifndef GEOMETRY_CALOGEOMETRY_IDEALOBLIQUEPRISM_H
#define GEOMETRY_CALOGEOMETRY_IDEALOBLIQUEPRISM_H 1

#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

  /** \class IdealObliquePrism
    
  Oblique prism class used for HCAL  (HB, HE, HO) volumes.
  
  Required parameters for an ideal oblique prism:
  
  - eta, phi of axis
  - radial distance (along axis) to front and back faces
  - single bit - faces parallel or perpendicular to z-axis
  - eta width and phi width of faces (same for front/back)
  
  Total: 6+1 parameters
  
  Internally, the "point of reference" is the center (eta/phi) of the
  front face of the prism.  Therefore, the only internally stored
  parameters are eta and phi widths, the axis tower thickness, and the
  parallel/perpendicular setting.  The parallel/perpendicular setting
  is encoded in the sign of the thickness.  (positive = parallel to
  z-axis, negative = perpendicular)

  \author J. Mans - Minnesota
  */
class IdealObliquePrism : public CaloCellGeometry 
{
public:

  typedef CaloCellGeometry::CCGFloat CCGFloat ;
  typedef CaloCellGeometry::Pt3D     Pt3D     ;
  typedef CaloCellGeometry::Pt3DVec  Pt3DVec  ;

  static constexpr uint32_t k_dEta = 0;//Eta-width
  static constexpr uint32_t k_dPhi = 1;//Phi-width
  static constexpr uint32_t k_dZ   = 2;//Signed thickness
  static constexpr uint32_t k_Eta  = 3;//Eta of the reference point
  static constexpr uint32_t k_Z    = 4;//Z   of the reference point

  IdealObliquePrism() ;
  IdealObliquePrism( const IdealObliquePrism& idop ) ;

  IdealObliquePrism& operator=( const IdealObliquePrism& idop ) ;
	 
  IdealObliquePrism( const GlobalPoint& faceCenter, 
		     CornersMgr*        mgr       ,
		     const CCGFloat*    parm       ) ;

  ~IdealObliquePrism() override ;

  CCGFloat dEta() const ;
  CCGFloat dPhi() const ;
  CCGFloat dz()   const ;
  CCGFloat eta()  const ;
  CCGFloat z()    const ;

  static void localCorners( Pt3DVec&        vec ,
			    const CCGFloat* pv  ,
			    Pt3D&           ref  ) ;

  void vocalCorners( Pt3DVec&        vec ,
			     const CCGFloat* pv  ,
			     Pt3D&           ref  ) const override;

private:
  void initCorners(CornersVec&)  override;

  static GlobalPoint etaPhiPerp( float eta, float phi, float perp ) ;
  static GlobalPoint etaPhiZ(float eta, float phi, float z) ;
};

std::ostream& operator<<( std::ostream& s , const IdealObliquePrism& cell ) ;

#endif
