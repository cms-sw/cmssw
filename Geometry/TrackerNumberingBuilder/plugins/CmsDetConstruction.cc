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

template <>
template <auto fxn>
void CmsDetConstruction<cms::DDFilteredView>::buildLoop(cms::DDFilteredView& fv,
                                                        GeometricDet* det,
                                                        const std::string& attribute){
  edm::LogVerbatim("TrackerNumberingBuilder") << "CmsDetConstruction::buildLoop start " << fv.geoHistory() <<std::endl;
  while(fv.firstChild()){
    edm::LogVerbatim("TrackerNumberingBuilder") << "buildLoop: " << fv.geoHistory() << std::endl
              << "\t HistorySize: " << fv.geoHistory().size() << std::endl;
    (this->*fxn)(fv,det,attribute);
  }
  edm::LogVerbatim("TrackerNumberingBuilder") << "CmsDetConstruction::buildLoop Exited Loop\n"
            << "\tPointer at: " << fv.geoHistory() <<std::endl;
}

template <>
template <auto fxn>
void CmsDetConstruction<DDFilteredView>::buildLoop(DDFilteredView& fv,
                                                   GeometricDet* det,
                                                   const std::string& attribute){
  edm::LogVerbatim("TrackerNumberingBuilder") << "CmsDetConstruction::buildLoop start " << fv.geoHistory() <<std::endl;
  bool dodets = fv.firstChild();  // descend to the first Layer
  while (dodets) {
    edm::LogVerbatim("TrackerNumberingBuilder") << "buildLoop: " << fv.geoHistory() << std::endl
                                                << "\t HistorySize: " << fv.geoHistory().size() << std::endl;
    (this->*fxn)(fv, det, attribute);
    dodets = fv.nextSibling();
  }
  edm::LogVerbatim("TrackerNumberingBuilder") << "CmsDetConstruction::buildLoop Exited Loop\n"
            << "\tPointer at: " << fv.geoHistory() <<std::endl;
}

template <class FilteredView>
void CmsDetConstruction<FilteredView>::buildComponent(FilteredView& fv,
                                                      GeometricDet* mother,
                                                      const std::string& attribute) {

  edm::LogVerbatim("TrackerNumberingBuilder") << "CmsDetConstruction::buildComponent " << fv.geoHistory() << std::endl;
  GeometricDet* det = new GeometricDet(&fv,
                                       CmsTrackerLevelBuilder<FilteredView>::theCmsTrackerStringToEnum.type(
                                           ExtractStringFromDDD<FilteredView>::getString(attribute, &fv)));

  //Phase1 mergedDet: searching for sensors
  if (CmsTrackerLevelBuilder<FilteredView>::theCmsTrackerStringToEnum.type(
          ExtractStringFromDDD<FilteredView>::getString(attribute, &fv)) == GeometricDet::mergedDet) {
    buildLoop<&CmsDetConstruction<FilteredView>::buildSmallDetsforGlued>(fv,det,attribute);

  }

  //Phase2 stackDet: same procedure, different nomenclature
  else if (CmsTrackerLevelBuilder<FilteredView>::theCmsTrackerStringToEnum.type(
               ExtractStringFromDDD<FilteredView>::getString(attribute, &fv)) == GeometricDet::OTPhase2Stack) {
    buildLoop<&CmsDetConstruction<FilteredView>::buildSmallDetsforStack>(fv,det,attribute);

  }

  mother->addComponent(det);
  edm::LogVerbatim("TrackerNumberingBuilder") << "\tMotherSize: " << mother->components().size() << std::endl;
  for(const auto& cp: mother->components()){
    edm::LogVerbatim("TrackerNumberingBuilder") << "\t" << cp->geographicalID().rawId() << std::endl;
  }
}

template class CmsDetConstruction<DDFilteredView>;
template class CmsDetConstruction<cms::DDFilteredView>;
