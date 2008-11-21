#include "CondDBCmsTrackerConstruction.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "CondFormats/IdealGeometryObjects/interface/PGeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerBuilder.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerDetIdBuilder.h"

using namespace cms;

CondDBCmsTrackerConstruction::CondDBCmsTrackerConstruction() { }

const GeometricDet* CondDBCmsTrackerConstruction::construct(const PGeometricDet& pgd) {
  
  GeometricDet* tracker  = new GeometricDet(pgd.pgeomdets_[0],GeometricDet::Tracker);
  // could do something like...  if ( tracker.type() != GeometricDet::Tracker ) {

  size_t detMax =  pgd.pgeomdets_.size();
  for (size_t tri = 1; tri < detMax; ++tri) {
    GeometricDet* subdet = new GeometricDet(pgd.pgeomdets_[tri], GeometricDet::GDEnumType(pgd.pgeomdets_[tri]._type));
    tracker->addComponent(subdet);
//     switch ( GeometricDet::GDEnumType(pgd.pgeomdets_[tri]._type) ) {
//     case GeometricDet::PixelBarrel:
      //      theCmsTrackerSubStrctBuilder.build(subdet);
      //      constructSubDet( pgd, subdet, tri );
    //    GeometricDet::GDEnumType currType(101);
//     do {
//       //      GeometricDet* subsubdet = new GeometricDet(pgd.pgeomdets_[tri], GeometricDet::GDEnumType(pgd.pgeomdets_[tri]._type));
//     } while ( 1 == 0);

//       break;
//     case GeometricDet::PixelEndCap:
//       theCmsTrackerSubStrctBuilder.build(fv,subdet,s);
//       break;
//     case GeometricDet::TIB:
//       theCmsTrackerSubStrctBuilder.build(fv,subdet,s);
//       break;
//     case GeometricDet::TOB:
//       theCmsTrackerSubStrctBuilder.build(fv,subdet,s);
//       break;
//     case GeometricDet::TEC:
//       theCmsTrackerSubStrctBuilder.build(fv,subdet,s);
//       break;
//     case GeometricDet::TID:
//       theCmsTrackerSubStrctBuilder.build(fv,subdet,s);
//       break;
//     default:
//       edm::LogError("CmsTrackerBuilder")<<" ERROR - I was expecting a SubDet, I got a "<<ExtractStringFromDDD::getString(s,&fv);
//       ;
//     }
  }
 
  return tracker;
}

// void CondDBCmsTrackerConstruction::constructSubDet( const PGeometricDet& pgd, size_t currentIndex ) {

//   size_t tri2;
//   while ( 

// }
