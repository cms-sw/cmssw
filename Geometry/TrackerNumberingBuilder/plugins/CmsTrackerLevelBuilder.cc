#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


void CmsTrackerLevelBuilder::build (
				    DDFilteredView& fv, 
				    GeometricDet* tracker,
				    std::string attribute){

  LogDebug("GeometricDetBuilding");
  LogTrace("GeometricDetBuilding") << "GeometricDet name and type: " << tracker->name() << " " << tracker->type() << std::endl;
  LogTrace("GeometricDetBuilding") << "Attribute string: " << attribute << std::endl;
  LogTrace("GeometricDetBuilding") << "Filtered View logical parts: " << fv.logicalPart().ddname().name() << std::endl;
  LogTrace("GeometricDetBuilding") << "Extracted string: " << ExtractStringFromDDD::getString(attribute,&fv) << std::endl;

 bool doLayers = fv.firstChild(); // descend to the first Layer  

  while (doLayers) {
    buildComponent(fv,tracker,attribute);      
    doLayers = fv.nextSibling(); // go to next layer
  }

  fv.parent();

 sortNS(fv,tracker);
}
