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
