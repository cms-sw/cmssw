#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"


void CmsTrackerLevelBuilder::build (
				    DDFilteredView& fv, 
				    GeometricDet* tracker,
				    std::string attribute){

 bool doLayers = fv.firstChild(); // descend to the first Layer  

  while (doLayers) {
    buildComponent(fv,tracker,attribute);      
    doLayers = fv.nextSibling(); // go to next layer
  }

  fv.parent();

 sortNS(fv,tracker);
}
