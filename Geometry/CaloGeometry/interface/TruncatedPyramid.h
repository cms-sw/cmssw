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

class TruncatedPyramid : public CaloCellGeometry
{
   public:

      typedef std::vector< HepPlane3D > BoundaryVec ;

      TruncatedPyramid( const CornersVec& corn  ) :
	 CaloCellGeometry ( corn ) , 
	 m_axis           ( axis() ) ,
	 m_bou            ( 0      ) {}

      virtual ~TruncatedPyramid() { delete m_bou ; }

      virtual bool inside( const GlobalPoint& point ) const ;  

      /** Position corresponding to the center of the front face at a certain
	  depth (default is zero) along the crystal axis.
	  If "depth" is <=0, the nomial position of the cell is returned
	  (center of the front face).
      */
      const GlobalPoint getPosition( float depth ) const 
      { return CaloCellGeometry::getPosition() + depth*m_axis ; }

      virtual const CornersVec& getCorners()       const { return CaloCellGeometry::getCorners() ; }

      // Return thetaAxis polar angle of axis of the crystal
      float getThetaAxis()                         const { return m_axis.theta() ; } 

      // Return phiAxis azimuthal angle of axis of the crystal
      float getPhiAxis()                           const { return m_axis.phi() ; } 

      const GlobalVector& axis()                   const { return m_axis ; }


      // for geometry creation in other classes
      static void createCorners( const std::vector<double>&  pv ,
				 const HepTransform3D&       tr ,
				 CornersVec&                 co   ) ;

      /// print out the element, with an optional string prefix, maybe OVAL identifier
      // why is this here with operator<< also? void dump( const char * prefix = "" ) const ;
   private:

      GlobalVector axis() 
      { 
	 return GlobalVector( backCtr() -
			      CaloCellGeometry::getPosition() ).unit() ;
      }

      const GlobalPoint backCtr() const 
      {
	 return GlobalPoint( 0.25*( getCorners()[4].x() + getCorners()[5].x() +
				    getCorners()[6].x() + getCorners()[7].x() ),
			     0.25*( getCorners()[4].y() + getCorners()[5].y() +
				    getCorners()[6].y() + getCorners()[7].y() ),
			     0.25*( getCorners()[4].z() + getCorners()[5].z() +
				    getCorners()[6].z() + getCorners()[7].z() ) ) ;
      }
      
      GlobalVector         m_axis ;

      mutable BoundaryVec* m_bou  ;
};

std::ostream& operator<<( std::ostream& s, const TruncatedPyramid& cell ) ;
  
#endif
