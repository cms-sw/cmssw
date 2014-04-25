#include "Geometry/CaloTopology/src/ShashlikGeometryBuilderFromDDD.h"
#include "Geometry/CaloTopology/interface/ShashlikGeometry.h"

#include <DetectorDescription/Core/interface/DDFilter.h>
#include <DetectorDescription/Core/interface/DDFilteredView.h>
#include <DetectorDescription/Core/interface/DDSolid.h>
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

ShashlikGeometryBuilderFromDDD::ShashlikGeometryBuilderFromDDD( void )
{}

ShashlikGeometryBuilderFromDDD::~ShashlikGeometryBuilderFromDDD( void )
{}

ShashlikGeometry*
ShashlikGeometryBuilderFromDDD::build( const DDCompactView* cview, const ShashlikTopology& topology ) 
{
  std::string attribute = "ReadOutName";
  std::string value     = "EcalHitsEK";
  DDValue val( attribute, value, 0.0 );
 
  // Asking only for the Shashlik's
  DDSpecificsFilter filter;
  filter.setCriteria( val, // name & value of a variable 
		      DDSpecificsFilter::matches,
		      DDSpecificsFilter::AND, 
		      true, // compare strings otherwise doubles
		      true // use merged-specifics or simple-specifics
    );
  DDFilteredView fview( *cview );
  fview.addFilter( filter );
  
  return this->buildGeometry( fview, topology );
}

ShashlikGeometry*
ShashlikGeometryBuilderFromDDD::buildGeometry( DDFilteredView& fview, const ShashlikTopology& topology )
{
  ShashlikGeometry* geometry = new ShashlikGeometry( topology );

  bool doSubDets = fview.firstChild();
 
  while( doSubDets )
  {
    std::cout << fview.logicalPart().name().name() << "\n";    
    int detid = 0;
 
    EKDetId ekid( detid );
    
//     std::vector<float> pv;
//     std::vector<GlobalPoint> corners( 8 );
//     const GlobalPoint front ( 0.25*( corners[0].x() + 
//                                      corners[1].x() + 
//                                      corners[2].x() + 
//                                      corners[3].x()   ),
//                               0.25*( corners[0].y() + 
//                                      corners[1].y() + 
//                                      corners[2].y() + 
//                                      corners[3].y()   ),
//                               0.25*( corners[0].z() + 
//                                      corners[1].z() + 
//                                      corners[2].z() + 
//                                      corners[3].z()   ) ) ;

//     const GlobalPoint back  ( 0.25*( corners[4].x() + 
// 				     corners[5].x() + 
//                                      corners[6].x() + 
//                                      corners[7].x()   ),
//                               0.25*( corners[4].y() + 
//                                      corners[5].y() + 
//                                      corners[6].y() + 
//                                      corners[7].y()   ),
//                               0.25*( corners[4].z() + 
//                                      corners[5].z() + 
//                                      corners[6].z() + 
//                                      corners[7].z()   ) ) ;
    
//     const CCGFloat* parmPtr ( CaloCellGeometry::getParmPtr( pv, 
// 							    geometry->parMgr(), 
//                                                             geometry->parVecVec()));
//     geometry->newCell( front, back, corners[0],
// 		       parmPtr, 
// 		       ekid ) ;
    
    doSubDets = fview.nextSibling(); // go to next layer
  }

  return geometry;
}
