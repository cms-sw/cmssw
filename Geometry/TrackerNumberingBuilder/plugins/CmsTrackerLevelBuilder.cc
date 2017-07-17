#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


void CmsTrackerLevelBuilder::build (
				    DDFilteredView& fv, 
				    GeometricDet* tracker,
				    std::string attribute){

  LogTrace("GeometricDetBuilding") << std::string(3*fv.history().size(),'-') 
				   << "+ "
				   << ExtractStringFromDDD::getString(attribute,&fv) << " " 
				   << tracker->type() << " " 
				   << tracker->name() 
				   << std::endl;

 bool doLayers = fv.firstChild(); // descend to the first Layer  

  while (doLayers) {
    buildComponent(fv,tracker,attribute);      
    doLayers = fv.nextSibling(); // go to next layer
  }

  fv.parent();

 sortNS(fv,tracker);
}
