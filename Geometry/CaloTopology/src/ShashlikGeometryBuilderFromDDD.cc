#include "ShashlikGeometryBuilderFromDDD.h"
#include "Geometry/CaloTopology/interface/ShashlikGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"

typedef CaloCellGeometry::CCGFloat CCGFloat;
typedef std::vector<float> ParmVec;
const double k_ScaleFromDDDtoGeant = 0.1;

ShashlikGeometryBuilderFromDDD::ShashlikGeometryBuilderFromDDD () {}
ShashlikGeometryBuilderFromDDD::~ShashlikGeometryBuilderFromDDD () {}


ShashlikGeometry* ShashlikGeometryBuilderFromDDD::build (const DDCompactView*  cpv, const ShashlikTopology& topology ) {
  // allocate geometry
  ShashlikGeometry* geom = new ShashlikGeometry (topology);
  unsigned int numberOfCells = (unsigned) topology.totalModules () * 2; // both sides
  geom->allocateCorners( numberOfCells ) ;
  geom->allocatePar( ShashlikGeometry::k_NumberOfParametersPerShape*ShashlikGeometry::k_NumberOfShapes,
		     ShashlikGeometry::k_NumberOfParametersPerShape ) ;

  // loop over modules
  DDFilteredView fv ( *cpv ) ;
  DDSpecificsFilter filter;
  filter.setCriteria( DDValue( "ShashlikStructure",
			       "ShashlikModule",
			       0                        ),
		      DDSpecificsFilter::equals,
		      DDSpecificsFilter::AND,
		      true,
		      true                               );
  fv.addFilter(filter);

  bool doSubDets = fv.firstChild();

  unsigned int counter = 0;
  while (doSubDets) {
    EKDetId detId( topology.denseId2detId(counter));
    ++counter ;
	
	
    DD3Vector x, y, z;
    fv.rotation().GetComponents( x, y, z ) ;
    const CLHEP::HepRep3x3 rotation ( x.X(), y.X(), z.X(),
				      x.Y(), y.Y(), z.Y(),
				      x.Z(), y.Z(), z.Z() );
    const CLHEP::HepRotation hr ( rotation );
    const CLHEP::Hep3Vector h3v ( fv.translation().X(),
				  fv.translation().Y(),
				  fv.translation().Z()  ) ;
    const HepGeom::Transform3D ht3d ( hr,          // only scale translation
				      k_ScaleFromDDDtoGeant*h3v ) ;    
	
    //	fillGeom( geom, parameters, ht3d, detId ) ;
    const DDSolid& solid ( fv.logicalPart().solid() ) ;
    ParmVec params (solid.parameters().begin(), solid.parameters().end());
    for (size_t i=0 ; i < params.size() ; ++i) {
      if (i==1 || i==2 || i==6 || i==10) params[i] *= k_ScaleFromDDDtoGeant;
    }

    std::vector<GlobalPoint> corners (8);

    TruncatedPyramid::createCorners( params, ht3d, corners ) ;

    const CCGFloat* parmPtr ( CaloCellGeometry::getParmPtr( params, 
							    geom->parMgr(), 
							    geom->parVecVec() ) ) ;
	
    GlobalPoint front ( 0.25*( corners[0].x() + 
			       corners[1].x() + 
			       corners[2].x() + 
			       corners[3].x()   ),
			0.25*( corners[0].y() + 
			       corners[1].y() + 
			       corners[2].y() + 
			       corners[3].y()   ),
			0.25*( corners[0].z() + 
			       corners[1].z() + 
			       corners[2].z() + 
			       corners[3].z()   ) ) ;
	
    GlobalPoint back  ( 0.25*( corners[4].x() + 
			       corners[5].x() + 
			       corners[6].x() + 
			       corners[7].x()   ),
			0.25*( corners[4].y() + 
			       corners[5].y() + 
			       corners[6].y() + 
			       corners[7].y()   ),
			0.25*( corners[4].z() + 
			       corners[5].z() + 
			       corners[6].z() + 
			       corners[7].z()   ) ) ;
	
    if (front.mag2() > back.mag2()) { // front should always point to the center, so swap front and back
      std::swap (front, back);
      std::swap_ranges (corners.begin(), corners.begin()+4, corners.begin()+4); 
    }
	
    geom->newCell( front, back, corners[0],
		   parmPtr, 
		   detId ) ;

    doSubDets = fv.nextSibling(); // go to next layer
  }
    
  assert( counter == numberOfCells ) ;
  
  geom->initializeParms() ;
  return geom;
}

