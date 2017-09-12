#include "CSCGeometryBuilderFromDDD.h"
#include "CSCGeometryBuilder.h"
#include "CSCGeometryParsFromDD.h"

#include <CondFormats/GeometryObjects/interface/RecoIdealGeometry.h>
#include <CondFormats/GeometryObjects/interface/CSCRecoDigiParameters.h>

#include <FWCore/Utilities/interface/Exception.h>

#include <utility>


CSCGeometryBuilderFromDDD::CSCGeometryBuilderFromDDD() : myName("CSCGeometryBuilderFromDDD"){}


CSCGeometryBuilderFromDDD::~CSCGeometryBuilderFromDDD(){}


void CSCGeometryBuilderFromDDD::build(std::shared_ptr<CSCGeometry> geom, const DDCompactView* cview, const MuonDDDConstants& muonConstants){

  RecoIdealGeometry rig;
  CSCRecoDigiParameters rdp;

  // simple class just really a method to get the parameters... but I want this method
  // available to classes other than CSCGeometryBuilderFromDDD so... simple class...
  CSCGeometryParsFromDD cscp;
  if ( ! cscp.build(cview, muonConstants, rig, rdp) ) {
    throw cms::Exception("CSCGeometryBuilderFromDDD", "Failed to build the necessary objects from the DDD");
  }
  CSCGeometryBuilder realbuilder;
  realbuilder.build(std::move(geom), rig, rdp);
  //  return realbuilder.build(rig, rdp); 

}
