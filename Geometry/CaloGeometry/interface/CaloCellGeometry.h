#ifndef GEOMETRY_CALOGEOMETRY_CALOCELLGEOMETRY_H
#define GEOMETRY_CALOGEOMETRY_CALOCELLGEOMETRY_H 1

#include "Geometry/CaloGeometry/interface/EZArrayFL.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include <vector>
#include <string>

/** \class CaloCellGeometry

Abstract base class for an individual cell's geometry.
    
$Date: 2007/09/07 19:08:33 $
$Revision: 1.10 $
\author J. Mans, P. Meridiani
*/

class CaloCellGeometry 
{
   public:

      typedef EZArrayFL< GlobalPoint > CornersVec ;
      typedef EZMgrFL< GlobalPoint >   CornersMgr ;

      typedef std::vector<float>  ParVec ;
      typedef std::vector<ParVec> ParVecVec ;

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


      static const float* getParmPtr( const std::vector<double>& vd ,
				      const unsigned int         np ,
				      ParVecVec&                 pvv ) ;

   protected:

      CaloCellGeometry( CornersVec::const_reference gp ,
			const CornersMgr*           mgr  ) :
	 m_refPoint ( gp  ),
	 m_corners  ( mgr ) {}

      CaloCellGeometry( const CornersVec& cv ) : 
	 m_refPoint ( GlobalPoint( 0.25*( cv[0].x() + cv[1].x() + cv[2].x() + cv[3].x() ),
				   0.25*( cv[0].y() + cv[1].y() + cv[2].y() + cv[3].y() ),
				   0.25*( cv[0].z() + cv[1].z() + cv[2].z() + cv[3].z() )  ) ), 
	 m_corners  ( cv ) {}

      CornersVec& setCorners() const { return m_corners ; }

   private:

      const   GlobalPoint m_refPoint ;

      mutable CornersVec  m_corners ;
};

std::ostream& operator<<( std::ostream& s, const CaloCellGeometry& cell ) ;

#endif
