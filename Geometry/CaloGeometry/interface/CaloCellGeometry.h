#ifndef GEOMETRY_CALOGEOMETRY_CALOCELLGEOMETRY_H
#define GEOMETRY_CALOGEOMETRY_CALOCELLGEOMETRY_H 1

#include "Geometry/CaloGeometry/interface/EZArrayFL.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include <CLHEP/Geometry/Point3D.h>
#include <CLHEP/Geometry/Transform3D.h>
#include <vector>
#include <string>
#include <cassert>

#include "FWCore/Utilities/interface/GCC11Compatibility.h"



/** \class CaloCellGeometry

Abstract base class for an individual cell's geometry.

The base class declares a pure virtual function and also writes a definition (body)
to force conscious acceptance of default behaviour.

If a derived class doesn't choose to override a normal virtual,
it just inherits the base version's behaviour by default. If you want
to provide a default behaviour but not let derived classes just inherit
it "silently" like this, you can make it pure virtual but still provide
a default that the derived class author has to call deliberately if he wants it:

@code
   class B {
     public:
         virtual bool f() = 0;
     };

     bool B::f() {
         return true;  // this is a good default, but
     }                 // shouldn't be used blindly

     class D : public B {
     public:
         bool f() {
             return B::f(); // if D wants the default
         }                  // behaviour, it has to say so
    };
@endcode

\author J. Mans, P. Meridiani
*/

class CaloCellGeometry 
{
public:

  typedef float                      CCGFloat ;
  typedef HepGeom::Transform3D       Tr3D     ;
  typedef HepGeom::Point3D<CCGFloat> Pt3D     ;
  typedef std::vector<Pt3D>          Pt3DVec  ;

  typedef EZArrayFL< GlobalPoint > CornersVec ;
  typedef EZMgrFL< GlobalPoint >   CornersMgr ;

  typedef EZArrayFL<CCGFloat> ParVec ;
  typedef std::vector<ParVec> ParVecVec ;
  typedef EZMgrFL< CCGFloat > ParMgr ;

  enum CornersSize { k_cornerSize = 8 };

  static const CCGFloat k_ScaleFromDDDtoGeant ;

  virtual ~CaloCellGeometry() ;
      
  /// Returns the corner points of this cell's volume.
  const CornersVec& getCorners() const { assert(not m_corners.uninitialized()); return m_corners; }

  /// Returns the position of reference for this cell 
  const GlobalPoint& getPosition() const {return m_refPoint;}
  const GlobalPoint& getBackPoint() const {return m_backPoint;} 

  float etaPos() const { return m_eta;}
  float phiPos() const { return m_phi;}

  float etaSpan() const { return m_dEta;}
  float	phiSpan() const	{ return m_dPhi;}


  /// Returns true if the specified point is inside this cell
  bool inside( const GlobalPoint & point ) const ;  

  bool emptyCorners() const { return m_corners.empty() ;}

  const CCGFloat* param() const { return m_parms ;}

  static const CCGFloat* checkParmPtr( const std::vector<CCGFloat>& vd  ,
				       ParVecVec&                   pvv   ) ;

  static const CCGFloat* getParmPtr( const std::vector<CCGFloat>& vd  ,
				     ParMgr*                      mgr ,
				     ParVecVec&                   pvv   ) ;


  ///----------- only needed by specific utility; overloaded when needed ----
  virtual void getTransform( Tr3D& tr, Pt3DVec* lptr ) const ;
  //------------------------------------------------------------------------

  virtual void vocalCorners( Pt3DVec&        vec ,
			     const CCGFloat* pv  ,
			     Pt3D&           ref  ) const = 0 ;

protected:

  CaloCellGeometry( CornersVec::const_reference gp ,
		    CornersMgr*                 mgr,
		    const CCGFloat*             par ) ;

  CaloCellGeometry( const CornersVec& cv,
		    const CCGFloat*   par ) ;

  CaloCellGeometry( void );

  // MUST be called by children constructors
  void initSpan() {
     initCorners(m_corners);
     m_dEta = std::abs(getCorners()[0].eta()-
                      getCorners()[2].eta());
     m_dPhi = std::abs(getCorners()[0].phi() -
                      getCorners()[2].phi());
     initBack();
  }

  virtual void initCorners(CornersVec&) = 0;
private:
 void initBack() {
    // from CaloTower code
    CornersVec const & cv = getCorners();
    m_backPoint = GlobalPoint(0.25 * (cv[4].x() + cv[5].x() + cv[6].x() + cv[7].x()),
                              0.25 * (cv[4].y() + cv[5].y() + cv[6].y() + cv[7].y()),
                              0.25 * (cv[4].z() + cv[5].z() + cv[6].z() + cv[7].z()));   
  }


  GlobalPoint         m_refPoint ;
  GlobalPoint         m_backPoint ;
  CornersVec  m_corners  ;
  const CCGFloat*     m_parms    ;
  float m_eta, m_phi;
  float m_dEta;
  float m_dPhi;

};

std::ostream& operator<<( std::ostream& s, const CaloCellGeometry& cell ) ;

#endif
