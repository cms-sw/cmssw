#include "Geometry/MTDNumberingBuilder/plugins/CmsMTDLevelBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDet.h"
#include "Geometry/MTDNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


void CmsMTDLevelBuilder::build (
				    DDFilteredView& fv, 
				    GeometricTimingDet* tracker,
				    std::string attribute){

  LogTrace("GeometricTimingDetBuilding") << std::string(3*fv.history().size(),'-') 
				   << "+ "
				   << ExtractStringFromDDD::getString(attribute,&fv) << " " 
				   << tracker->type() << " " 
				   << tracker->name() 
				   << std::endl;

 bool doLayers = fv.firstChild(); // descend to the next Layer  

  while (doLayers) {
    buildComponent(fv,tracker,attribute);      
    doLayers = fv.nextSibling(); // go to the next adjacent thingy
  }

  fv.parent();

 sortNS(fv,tracker);

}
