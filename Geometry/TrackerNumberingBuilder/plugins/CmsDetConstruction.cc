#include "Geometry/TrackerNumberingBuilder/plugins/CmsDetConstruction.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "DataFormats/DetId/interface/DetId.h"

void CmsDetConstruction::buildComponent(DDFilteredView& fv, 
					GeometricDet *mother, 
					std::string attribute){
  
  //
  // at this level I check whether it is a merged detector or not
  //
  LogTrace("DetConstruction") << " CmsDetConstruction::buildComponent ";

  GeometricDet * det  = new GeometricDet(&fv,theCmsTrackerStringToEnum.type(ExtractStringFromDDD::getString(attribute,&fv)));
  if (theCmsTrackerStringToEnum.type(ExtractStringFromDDD::getString(attribute,&fv)) ==  GeometricDet::mergedDet){
    //
    // I have to go one step lower ...
    //
    bool dodets = fv.firstChild(); // descend to the first Layer
    while (dodets) {
      buildSmallDetsforGlued(fv,det,attribute);
      dodets = fv.nextSibling(); // go to next layer
	/*
	Add algo to sort the merged DET
	*/
    }
    fv.parent();
  } else if (theCmsTrackerStringToEnum.type(ExtractStringFromDDD::getString(attribute,&fv)) ==  GeometricDet::OTPhase2Stack){
  
    LogTrace("DetConstruction") << " a stack ";

    bool dodets = fv.firstChild(); // descend to the first Layer
    while (dodets) {
      LogTrace("DetConstruction") << " new child! ";
      buildSmallDetsforStack(fv,det,attribute);
      dodets = fv.nextSibling(); // go to next layer
    }
    fv.parent();
  }

  
  mother->addComponent(det);
}

void CmsDetConstruction::buildSmallDetsforGlued(DDFilteredView& fv, 
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

void CmsDetConstruction::buildSmallDetsforStack(DDFilteredView& fv,
        	                                GeometricDet *mother,
                	                        std::string attribute){

  LogTrace("DetConstruction") << " CmsPhase2OTDetConstruction::buildSmallDetsforStacks ";
  GeometricDet * det  = new GeometricDet(&fv, theCmsTrackerStringToEnum.type(ExtractStringFromDDD::getString(attribute,&fv)));
  static const std::string isInner = "TrackerInnerDetectors";
  static const std::string isOuter = "TrackerOuterDetectors";
/*
  if (ExtractStringFromDDD::getString(isInner,&fv) == "true"){
    LogTrace("DetConstruction") << " inner ";
    uint32_t temp = 1;
    det->setGeographicalID(DetId(temp));
  } else if (ExtractStringFromDDD::getString(isOuter,&fv) == "true"){
    LogTrace("DetConstruction") << " outer ";
    uint32_t temp = 2;
    det->setGeographicalID(DetId(temp));
  } else {
    edm::LogError("DetConstruction") << " module defined as a Stack but not inner either outer!? ";
  }
*/
  mother->addComponent(det);
}
