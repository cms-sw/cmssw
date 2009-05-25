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

      typedef std::vector< HepGeom::Plane3D<double>  > BoundaryVec ;

      TruncatedPyramid( const CornersMgr*  cMgr ,
			const GlobalPoint& fCtr ,
			const GlobalPoint& bCtr ,
			const GlobalPoint& cor1 ,
			const double*      parV   ) :
	 CaloCellGeometry ( fCtr, cMgr, parV ) ,
	 m_axis           ( ( bCtr - fCtr ).unit() ) ,
	 m_corOne         ( new HepGeom::Point3D<double> ( cor1.x(), cor1.y(), cor1.z() ) )
      {} 

      TruncatedPyramid( const CornersVec& corn,
			const double*     par  ) :
	 CaloCellGeometry ( corn, par   ) , 
	 m_axis           ( makeAxis() ) ,
	 m_corOne         ( new HepGeom::Point3D<double> ( corn[0].x(),
					    corn[0].y(),
					    corn[0].z()  ) )    {} 

      virtual ~TruncatedPyramid() { delete m_corOne ; }

      virtual bool inside( const GlobalPoint& point ) const ;  

      /** Position corresponding to the center of the front face at a certain
	  depth (default is zero) along the crystal axis.
	  If "depth" is <=0, the nomial position of the cell is returned
	  (center of the front face).
      */
      const GlobalPoint getPosition( float depth ) const 
      { return CaloCellGeometry::getPosition() + depth*m_axis ; }

      virtual const CornersVec& getCorners()       const ;

      // Return thetaAxis polar angle of axis of the crystal
      float getThetaAxis()                         const { return m_axis.theta() ; } 

      // Return phiAxis azimuthal angle of axis of the crystal
      float getPhiAxis()                           const { return m_axis.phi() ; } 

      const GlobalVector& axis()                   const { return m_axis ; }


      // for geometry creation in other classes
      static void createCorners( const std::vector<double>&  pv ,
				 const HepGeom::Transform3D&       tr ,
				 CornersVec&                 co   ) ;

      virtual std::vector<HepGeom::Point3D<double> > vocalCorners( const double* pv,
						    HepGeom::Point3D<double> &   ref ) const
      { return localCorners( pv, ref ) ; }

      static std::vector<HepGeom::Point3D<double> > localCorners( const double* pv,
						   HepGeom::Point3D<double> &   ref ) ;
      static std::vector<HepGeom::Point3D<double> > localCornersReflection( const double* pv,
							     HepGeom::Point3D<double> &   ref ) ;

      static std::vector<HepGeom::Point3D<double> > localCornersSwap( const double* pv,
						       HepGeom::Point3D<double> &   ref ) ;

      virtual HepGeom::Transform3D getTransform( std::vector<HepGeom::Point3D<double> >* lptr ) const ;

   private:


      GlobalVector makeAxis() 
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

      mutable HepGeom::Point3D<double> *  m_corOne ;
};

std::ostream& operator<<( std::ostream& s, const TruncatedPyramid& cell ) ;
  
#endif
