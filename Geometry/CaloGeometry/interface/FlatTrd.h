#ifndef GeometryCaloGeometryFlatTrd_h
#define GeometryCaloGeometryFlatTrd_h

#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include <CLHEP/Geometry/Point3D.h>
#include <CLHEP/Geometry/Plane3D.h>
#include <CLHEP/Geometry/Vector3D.h>
#include <CLHEP/Geometry/Transform3D.h>
#include <vector>

/**

   \class FlatTrd

   \brief A base class to handle the particular shape of HGCal volumes.
   
*/

class FlatTrd : public CaloCellGeometry {
public:

  typedef CaloCellGeometry::CCGFloat CCGFloat ;
  typedef CaloCellGeometry::Pt3D     Pt3D     ;
  typedef CaloCellGeometry::Pt3DVec  Pt3DVec  ;
  typedef CaloCellGeometry::Tr3D     Tr3D     ;

  static constexpr uint32_t k_dZ   = 0;//Half-length along the z-axis
  static constexpr uint32_t k_Theta= 1;//Polar angle of the line joining the
                                       //centres of the faces at -/+ dZ
  static constexpr uint32_t k_Phi  = 2;//Azimuthal angle of the line joing the
                                       //centres of the faces at -/+ dZ
  static constexpr uint32_t k_dY1  = 3;//Half-length along y of the face at -dZ
  static constexpr uint32_t k_dX1  = 4;//Half-length along x of the side at 
                                       //y=-dY1 of the face at -dZ
  static constexpr uint32_t k_dX2  = 5;//Half-length along x of the side at 
                                       //y=+dY1 of the face at -dZ
  static constexpr uint32_t k_Alp1 = 6;//Angle w.r.t the y axis from the center
                                       //of the sides at y=-dY1 to at y=+dY1
  static constexpr uint32_t k_dY2  = 7;//Half-length along y of the face at +dZ
  static constexpr uint32_t k_dX3  = 8;//Half-length along x of the side at 
                                       //y=-dY2 of the face at +dZ
  static constexpr uint32_t k_dX4  = 9;//Half-length along x of the side at 
                                       //y=+dY2 of the face at +dZ
  static constexpr uint32_t k_Alp2 =10;//Angle w.r.t the y axis from the center
                                       //of the sides at y=-dY2 to at y=+dY2
  static constexpr uint32_t k_Cell =11;//Cell size 
  //Assumes dY2=dY1; dX3=dX1; dX4=dX2; Alp2=Alp1=Theta=Phi=0  

  FlatTrd( void );
  
  FlatTrd( const FlatTrd& tr ) ;
  
  FlatTrd& operator=( const FlatTrd& tr ) ;
  
  FlatTrd( CornersMgr*  cMgr ,
	   const GlobalPoint& fCtr ,
	   const GlobalPoint& bCtr ,
	   const GlobalPoint& cor1 ,
	   const CCGFloat*    parV   ) ;
  
  FlatTrd( const CornersVec& corn ,
	   const CCGFloat*   par    ) ;
  
  FlatTrd( const FlatTrd& tr, const Pt3D& local) ;

  ~FlatTrd() override ;
  
  GlobalPoint const & getPosition() const override { return m_global; }
  GlobalPoint getPosition( const Pt3D& local ) const override;
  virtual float etaPos() const { return m_global.eta(); }
  virtual float phiPos() const { return m_global.phi(); }
  Pt3D getLocal( const GlobalPoint& global ) const;

  // Return thetaAxis polar angle of axis of the crystal
  CCGFloat getThetaAxis() const ;
  
  // Return phiAxis azimuthal angle of axis of the crystal
  CCGFloat getPhiAxis() const ;

  void vocalCorners( Pt3DVec&        vec ,
		     const CCGFloat* pv  ,
		     Pt3D&           ref  ) const override;
  
  const GlobalVector& axis() const ;
  
  static void createCorners( const std::vector<CCGFloat>& pv ,
                             const Tr3D&                  tr ,
                             std::vector<GlobalPoint>&    co   ) ;
  
  static void localCorners( Pt3DVec&        vec ,
                            const CCGFloat* pv  ,
                            Pt3D&           ref  ) ;
  
  void getTransform( Tr3D& tr, Pt3DVec* lptr ) const override;

  void setPosition ( const GlobalPoint& p ) { m_global = p;  setRefPoint(p); }

  static constexpr unsigned int ncorner_    = 8;
  static constexpr unsigned int ncornerBy2_ = 4;

private:

  void initCorners(CornersVec& ) override;
  
  GlobalVector makeAxis( void );
  
  GlobalPoint backCtr( void ) const;    
  GlobalVector m_axis;
  Pt3D         m_corOne, m_local;
  GlobalPoint  m_global;
  Tr3D         m_tr;
};

std::ostream& operator<<( std::ostream& s, const FlatTrd& cell ) ;

#endif
