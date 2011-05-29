#ifndef GEOMETRY_CALOGEOMETRY_CALOCELLGEOMETRY_H
#define GEOMETRY_CALOGEOMETRY_CALOCELLGEOMETRY_H 1

#include "Geometry/CaloGeometry/interface/EZArrayFL.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include <CLHEP/Geometry/Point3D.h>
#include <CLHEP/Geometry/Transform3D.h>
#include <vector>
#include <string>

/** \class CaloCellGeometry

Abstract base class for an individual cell's geometry.
    
$Date: 2010/04/20 17:23:11 $
$Revision: 1.18 $
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
      
      // Returns the corner points of this cell's volume
      virtual const CornersVec& getCorners() const = 0 ;

      // Returns the position of reference for this cell 
      const GlobalPoint& getPosition() const ;

      // Returns true if the specified point is inside this cell
      bool inside( const GlobalPoint & point ) const ;  

      bool emptyCorners() const ;

      const CCGFloat* param() const ;

      static const CCGFloat* checkParmPtr( const std::vector<CCGFloat>& vd  ,
					   ParVecVec&                   pvv   ) ;

      static const CCGFloat* getParmPtr( const std::vector<CCGFloat>& vd  ,
					 ParMgr*                      mgr ,
					 ParVecVec&                   pvv   ) ;


//----------- only needed by specific utility; overloaded when needed ----
      virtual void getTransform( Tr3D& tr, Pt3DVec* lptr ) const ;
//------------------------------------------------------------------------

      virtual void vocalCorners( Pt3DVec&        vec ,
				 const CCGFloat* pv  ,
				 Pt3D&           ref  ) const = 0 ;

   protected:

      CaloCellGeometry( CornersVec::const_reference gp ,
			const CornersMgr*           mgr,
			const CCGFloat*             par ) ;

      CaloCellGeometry( const CornersVec& cv,
			const CCGFloat*   par ) ;

      CornersVec& setCorners() const ;

      CaloCellGeometry() ;
      CaloCellGeometry( const CaloCellGeometry& cell ) ;
      CaloCellGeometry& operator=( const CaloCellGeometry& cell ) ;

   private:

      GlobalPoint         m_refPoint ;

      mutable CornersVec  m_corners  ;

      const CCGFloat*     m_parms    ;
};

std::ostream& operator<<( std::ostream& s, const CaloCellGeometry& cell ) ;

#endif
