#ifndef TruncatedPyramid_h
#define TruncatedPyramid_h

#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include <CLHEP/Geometry/Point3D.h>
#include <CLHEP/Geometry/Plane3D.h>
#include <CLHEP/Geometry/Vector3D.h>
#include <CLHEP/Geometry/Transform3D.h>
#include <vector>

/**

   \class TruncatedPyramid

   \brief A base class to handle the particular shape of Ecal Xtals. Taken from ORCA Calorimetry Code
   
*/

class TruncatedPyramid  GCC11_FINAL : public CaloCellGeometry {
public:

  typedef CaloCellGeometry::CCGFloat CCGFloat ;
  typedef CaloCellGeometry::Pt3D     Pt3D     ;
  typedef CaloCellGeometry::Pt3DVec  Pt3DVec  ;
  typedef CaloCellGeometry::Tr3D     Tr3D     ;
  
  TruncatedPyramid( void );
  
  TruncatedPyramid( const TruncatedPyramid& tr ) ;
  
  TruncatedPyramid& operator=( const TruncatedPyramid& tr ) ;
  
  TruncatedPyramid( const CornersMgr*  cMgr ,
		    const GlobalPoint& fCtr ,
		    const GlobalPoint& bCtr ,
		    const GlobalPoint& cor1 ,
		    const CCGFloat*    parV   ) ;
  
  TruncatedPyramid( const CornersVec& corn ,
		    const CCGFloat*   par    ) ;
  
  virtual ~TruncatedPyramid() ;
  
  const GlobalPoint getPosition( CCGFloat depth ) const ;
  
  virtual const CornersVec& getCorners() const ;
  
  // Return thetaAxis polar angle of axis of the crystal
  CCGFloat getThetaAxis() const ;
  
  // Return phiAxis azimuthal angle of axis of the crystal
  CCGFloat getPhiAxis() const ;
  
  const GlobalVector& axis() const ;
  
  // for geometry creation in other classes
  static void createCorners( const std::vector<CCGFloat>& pv ,
			     const Tr3D&                  tr ,
			     std::vector<GlobalPoint>&    co   ) ;
  
  virtual void vocalCorners( Pt3DVec&        vec ,
			     const CCGFloat* pv  ,
			     Pt3D&           ref  ) const ;
  
  static void localCorners( Pt3DVec&        vec ,
			    const CCGFloat* pv  ,
			    Pt3D&           ref  ) ;
  
  static void localCornersReflection( Pt3DVec&        vec ,
				      const CCGFloat* pv  ,
				      Pt3D&           ref  ) ;
  
  static void localCornersSwap( Pt3DVec&        vec ,
				const CCGFloat* pv  ,
				Pt3D&           ref  ) ;
  
  virtual void getTransform( Tr3D& tr, Pt3DVec* lptr ) const ;
  
private:
  GlobalVector makeAxis( void );
  
  const GlobalPoint backCtr( void ) const;    
  GlobalVector m_axis;
  Pt3D         m_corOne;
};

std::ostream& operator<<( std::ostream& s, const TruncatedPyramid& cell ) ;

#endif
