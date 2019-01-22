#ifndef GEOMETRY_CALOGEOMETRY_IDEALZPRISM_H
#define GEOMETRY_CALOGEOMETRY_IDEALZPRISM_H 1

#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include <memory>

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
class IdealZPrism final : public CaloCellGeometry 
{
 public:

  enum DEPTH {None, EM, HADR};
  
      typedef CaloCellGeometry::CCGFloat CCGFloat ;
      typedef CaloCellGeometry::Pt3D     Pt3D     ;
      typedef CaloCellGeometry::Pt3DVec  Pt3DVec  ;

      static constexpr uint32_t k_dEta = 0;//Eta-width
      static constexpr uint32_t k_dPhi = 1;//Phi-width
      static constexpr uint32_t k_dZ   = 2;//Signed thickness
      static constexpr uint32_t k_Eta  = 3;//Eta of the reference point
      static constexpr uint32_t k_Z    = 4;//Z   of the reference point
      
      IdealZPrism() ;
      
      IdealZPrism( const IdealZPrism& idzp ) ;
      
      IdealZPrism& operator=( const IdealZPrism& idzp ) ;
      
      IdealZPrism( const GlobalPoint& faceCenter , 
		   CornersMgr*        mgr        ,
		   const CCGFloat*    parm       ,
			  IdealZPrism::DEPTH depth) ;
      
      ~IdealZPrism() override ;
      
      CCGFloat dEta() const ;
      CCGFloat dPhi() const ;
      CCGFloat dz()   const ;
      CCGFloat eta()  const ;
      CCGFloat z()    const ;
      
      static void localCorners( Pt3DVec&        vec ,
				const CCGFloat* pv  ,
				Pt3D&           ref   ) ;
      
      void vocalCorners( Pt3DVec&        vec ,
				 const CCGFloat* pv  ,
				 Pt3D&           ref   ) const override;


  
  
      // corrected geom for PF
      std::shared_ptr<const IdealZPrism>  forPF() const  { 
	static const auto do_not_delete = [](const void*){};
	auto cell = std::shared_ptr<const IdealZPrism>(m_geoForPF.get(),do_not_delete);
	return cell;
      }
  
   private:

      void initCorners(CornersVec& ) override;
      
      static GlobalPoint etaPhiR( float eta ,
				  float phi ,
				  float rad   ) ;

      static GlobalPoint etaPhiPerp( float eta , 
				     float phi , 
				     float perp  ) ;

      static GlobalPoint etaPhiZ( float eta , 
				  float phi ,
				  float z    ) ;


private:
      // corrected geom for PF
      std::unique_ptr<IdealZPrism> m_geoForPF;

};

std::ostream& operator<<( std::ostream& s , const IdealZPrism& cell ) ;

#endif
