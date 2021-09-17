/*
// \class CSCGeometryBuilderFromDDD
//
//  Description: CSC Geometry Builder for DD4hep
//              
//
// \author Sergio Lo Meo (sergio.lo.meo@cern.ch) following what Ianna Osburne made for DTs (DD4HEP migration)
//         Created:  Thu, 05 March 2020 
//         Modified: Thu, 04 June 2020, following what made in PR #30047               
//   
//         Original author: Tim Cox
*/
//
#include "CSCGeometryBuilderFromDDD.h"
#include "CSCGeometryBuilder.h"
#include "Geometry/CSCGeometryBuilder/interface/CSCGeometryParsFromDD.h"
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"
#include "CondFormats/GeometryObjects/interface/CSCRecoDigiParameters.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <utility>

CSCGeometryBuilderFromDDD::CSCGeometryBuilderFromDDD() : myName("CSCGeometryBuilderFromDDD") {}

CSCGeometryBuilderFromDDD::~CSCGeometryBuilderFromDDD() {}
// DDD
void CSCGeometryBuilderFromDDD::build(CSCGeometry& geom,
                                      const DDCompactView* cview,
                                      const MuonGeometryConstants& muonConstants) {
  RecoIdealGeometry rig;
  CSCRecoDigiParameters rdp;

  // simple class just really a method to get the parameters... but I want this method
  // available to classes other than CSCGeometryBuilderFromDDD so... simple class...
  CSCGeometryParsFromDD cscp;
  if (!cscp.build(cview, muonConstants, rig, rdp)) {
    throw cms::Exception("CSCGeometryBuilderFromDDD", "Failed to build the necessary objects from the DDD");
  }
  CSCGeometryBuilder realbuilder;
  realbuilder.build(geom, rig, rdp);
}

// for DD4hep

void CSCGeometryBuilderFromDDD::build(CSCGeometry& geom,
                                      const cms::DDCompactView* cview,
                                      const MuonGeometryConstants& muonConstants) {
  RecoIdealGeometry rig;
  CSCRecoDigiParameters rdp;

  CSCGeometryParsFromDD cscp;
  if (!cscp.build(cview, muonConstants, rig, rdp)) {
    throw cms::Exception("CSCGeometryBuilderFromDDD", "Failed to build the necessary objects from the DD4HEP");
  }

  CSCGeometryBuilder realbuilder;
  realbuilder.build(geom, rig, rdp);
}
