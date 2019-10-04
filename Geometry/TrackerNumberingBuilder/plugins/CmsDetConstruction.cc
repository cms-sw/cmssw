#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsDetConstruction.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "DataFormats/DetId/interface/DetId.h"

template <class FilteredView>
void CmsDetConstruction<FilteredView>::buildSmallDetsforGlued(FilteredView& fv,
                                                              GeometricDet* mother,
                                                              const std::string& attribute) {
  GeometricDet* det = new GeometricDet(&fv,
                                       CmsTrackerLevelBuilder<FilteredView>::theCmsTrackerStringToEnum.type(
                                           ExtractStringFromDDD<FilteredView>::getString(attribute, &fv)));
  static const std::string stereo = "TrackerStereoDetectors";

  if (ExtractStringFromDDD<FilteredView>::getString(stereo, &fv) == "true") {
    uint32_t temp = 1;
    det->setGeographicalID(DetId(temp));
  } else {
    uint32_t temp = 2;
    det->setGeographicalID(DetId(temp));
  }

  mother->addComponent(det);
}

template <class FilteredView>
void CmsDetConstruction<FilteredView>::buildSmallDetsforStack(FilteredView& fv,
                                                              GeometricDet* mother,
                                                              const std::string& attribute) {
  GeometricDet* det = new GeometricDet(&fv,
                                       CmsTrackerLevelBuilder<FilteredView>::theCmsTrackerStringToEnum.type(
                                           ExtractStringFromDDD<FilteredView>::getString(attribute, &fv)));
  static const std::string isLower = "TrackerLowerDetectors";
  static const std::string isUpper = "TrackerUpperDetectors";

  if (ExtractStringFromDDD<FilteredView>::getString(isLower, &fv) == "true") {
    uint32_t temp = 1;
    det->setGeographicalID(DetId(temp));
  } else if (ExtractStringFromDDD<FilteredView>::getString(isUpper, &fv) == "true") {
    uint32_t temp = 2;
    det->setGeographicalID(DetId(temp));
  } else {
    edm::LogError("DetConstruction") << " module defined in a Stack but not upper either lower!? ";
  }
  mother->addComponent(det);
}

template <class FilteredView>
void CmsDetConstruction<FilteredView>::buildComponent(FilteredView& fv,
                                                      GeometricDet* mother,
                                                      const std::string& attribute) {
  //
  // at this level I check whether it is a merged detector or not
  //

  GeometricDet* det = new GeometricDet(&fv,
                                       CmsTrackerLevelBuilder<FilteredView>::theCmsTrackerStringToEnum.type(
                                           ExtractStringFromDDD<FilteredView>::getString(attribute, &fv)));

  //Phase1 mergedDet: searching for sensors
  if (CmsTrackerLevelBuilder<FilteredView>::theCmsTrackerStringToEnum.type(
          ExtractStringFromDDD<FilteredView>::getString(attribute, &fv)) == GeometricDet::mergedDet) {
    // I have to go one step lower ...
    bool dodets = fv.firstChild();  // descend to the first Layer
    while (dodets) {
      buildSmallDetsforGlued(fv, det, attribute);
      dodets = fv.nextSibling();  // go to next layer
                                  /*
	Add algo to sort the merged DET
	*/
    }
    fv.parent();

  }

  //Phase2 stackDet: same procedure, different nomenclature
  else if (CmsTrackerLevelBuilder<FilteredView>::theCmsTrackerStringToEnum.type(
               ExtractStringFromDDD<FilteredView>::getString(attribute, &fv)) == GeometricDet::OTPhase2Stack) {
    bool dodets = fv.firstChild();
    while (dodets) {
      buildSmallDetsforStack(fv, det, attribute);
      dodets = fv.nextSibling();
    }
    fv.parent();
  }

  mother->addComponent(det);
}

template class CmsDetConstruction<DDFilteredView>;
template class CmsDetConstruction<cms::DDFilteredView>;
