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
    
$Date: 2009/01/23 15:03:24 $
$Revision: 1.16 $
\author J. Mans, P. Meridiani
*/





class CaloCellGeometry 
{
   public:

      typedef EZArrayFL< GlobalPoint > CornersVec ;
      typedef EZMgrFL< GlobalPoint >   CornersMgr ;

      typedef EZArrayFL<double>   ParVec ;
      typedef std::vector<ParVec> ParVecVec ;
      typedef EZMgrFL< double >   ParMgr ;

      enum CornersSize { k_cornerSize = 8 };

      static const float k_ScaleFromDDDtoGeant ;

      virtual ~CaloCellGeometry() {}
      
      // Returns the corner points of this cell's volume
      virtual const CornersVec& getCorners() const = 0 ;

      // Returns the position of reference for this cell 
      const GlobalPoint& getPosition() const { return m_refPoint ; }

      // Returns true if the specified point is inside this cell
      virtual bool inside( const GlobalPoint & point ) const = 0 ;  

      bool emptyCorners() const { return m_corners.empty() ; }

      const double* param() const { return m_parms ; }

      static const double* checkParmPtr( const std::vector<double>& vd  ,
					 ParVecVec&                 pvv ) ;

      static const double* getParmPtr( const std::vector<double>& vd  ,
				       ParMgr*                    mgr ,
				       ParVecVec&                 pvv ) ;


//----------- only needed by specific utility; overloaded when needed ----
      virtual HepGeom::Transform3D getTransform( std::vector<HepGeom::Point3D<double> >* lptr ) const ;
//------------------------------------------------------------------------

      virtual std::vector<HepGeom::Point3D<double> > vocalCorners( const double* pv,
						    HepGeom::Point3D<double> &   ref ) const = 0 ;

   protected:

      CaloCellGeometry( CornersVec::const_reference gp ,
			const CornersMgr*           mgr,
			const double*               par ) :
	 m_refPoint ( gp  ),
	 m_corners  ( mgr ),
	 m_parms    ( par ) {}

      CaloCellGeometry( const CornersVec& cv,
			const double*     par ) : 
	 m_refPoint ( GlobalPoint( 0.25*( cv[0].x() + cv[1].x() + cv[2].x() + cv[3].x() ),
				   0.25*( cv[0].y() + cv[1].y() + cv[2].y() + cv[3].y() ),
				   0.25*( cv[0].z() + cv[1].z() + cv[2].z() + cv[3].z() )  ) ), 
	 m_corners  ( cv ),
	 m_parms    ( par ) {}

      CornersVec& setCorners() const { return m_corners ; }

   private:

      const   GlobalPoint m_refPoint ;

      mutable CornersVec  m_corners ;

      const double*        m_parms  ;
};

std::ostream& operator<<( std::ostream& s, const CaloCellGeometry& cell ) ;

#endif
