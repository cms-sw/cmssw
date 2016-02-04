#include "Geometry/TrackerNumberingBuilder/plugins/CmsDetConstruction.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "DataFormats/DetId/interface/DetId.h"

void CmsDetConstruction::buildComponent(
					DDFilteredView& fv, 
					GeometricDet *mother, 
					std::string attribute){
  
  //
  // at this level I check whether it is a merged detector or not
  //

  GeometricDet * det  = new GeometricDet(&fv,theCmsTrackerStringToEnum.type(ExtractStringFromDDD::getString(attribute,&fv)));
  if (theCmsTrackerStringToEnum.type(ExtractStringFromDDD::getString(attribute,&fv)) ==  GeometricDet::mergedDet){
    //
    // I have to go one step lower ...
    //
    bool dodets = fv.firstChild(); // descend to the first Layer
    while (dodets) {
      buildSmallDets(fv,det,attribute);
      dodets = fv.nextSibling(); // go to next layer
	/*
	Add algo to sort the merged DET
	*/
    }
    fv.parent();
  }
  
  mother->addComponent(det);
}

void CmsDetConstruction::buildSmallDets( 
					DDFilteredView& fv, 
					GeometricDet *mother, 
					std::string attribute){

  GeometricDet * det  = 
    new GeometricDet(&fv,
		     theCmsTrackerStringToEnum.type(ExtractStringFromDDD::getString(attribute,&fv)));
  static const std::string stereo = "TrackerStereoDetectors";
  if (ExtractStringFromDDD::getString(stereo,&fv) == "true"){
    uint32_t temp = 1;
    det->setGeographicalID(DetId(temp));
  }else{
    uint32_t temp = 2;
    det->setGeographicalID(DetId(temp));
  }
  
  mother->addComponent(det); 
}

