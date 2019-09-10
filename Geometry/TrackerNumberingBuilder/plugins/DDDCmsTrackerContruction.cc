#include "Geometry/TrackerNumberingBuilder/plugins/DDDCmsTrackerContruction.h"

#include <utility>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerBuilder.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerDetIdBuilder.h"

std::unique_ptr<GeometricDet> DDDCmsTrackerContruction::construct(const DDCompactView& cpv,
                                                                  std::vector<int> const& detidShifts) {
  std::string attribute = "TkDDDStructure";
  DDSpecificsHasNamedValueFilter filter{attribute};

  DDFilteredView fv(cpv, filter);

  CmsTrackerStringToEnum theCmsTrackerStringToEnum;
  if (theCmsTrackerStringToEnum.type(ExtractStringFromDDD<DDFilteredView>::getString(attribute, &fv)) !=
      GeometricDet::Tracker) {
    fv.firstChild();
    if (theCmsTrackerStringToEnum.type(ExtractStringFromDDD<DDFilteredView>::getString(attribute, &fv)) !=
        GeometricDet::Tracker) {
      throw cms::Exception("Configuration") << " The first child of the DDFilteredView is not what is expected \n"
                                            << ExtractStringFromDDD<DDFilteredView>::getString(attribute, &fv) << "\n";
    }
  }

  auto tracker = std::make_unique<GeometricDet>(&fv, GeometricDet::Tracker);
  CmsTrackerBuilder<DDFilteredView> theCmsTrackerBuilder;
  theCmsTrackerBuilder.build(fv, tracker.get(), attribute);

  CmsTrackerDetIdBuilder theCmsTrackerDetIdBuilder(detidShifts);

  theCmsTrackerDetIdBuilder.buildId(*tracker);
  fv.parent();
  //
  // set the Tracker
  //
  //TrackerMapDDDtoID::instance().setTracker(tracker);
  //NOTE: If it is decided that the TrackerMapDDDtoID should be
  // constructed here, then we should return from this
  // function so that the EventSetup can manage it

  return tracker;
}

std::unique_ptr<GeometricDet> DDDCmsTrackerContruction::construct(const cms::DDCompactView& cpv,
                                                                  std::vector<int> const& detidShifts) {
  std::string attribute("TkDDDStructure");
  cms::DDFilteredView fv(cpv, cms::DDFilter(attribute));

  CmsTrackerStringToEnum theCmsTrackerStringToEnum;
  if (theCmsTrackerStringToEnum.type(ExtractStringFromDDD<cms::DDFilteredView>::getString("TkDDDStructure", &fv)) !=
      GeometricDet::Tracker) {
    fv.firstChild();
    if (theCmsTrackerStringToEnum.type(ExtractStringFromDDD<cms::DDFilteredView>::getString(attribute, &fv)) !=
        GeometricDet::Tracker) {
      throw cms::Exception("Configuration")
          << " The first child of the DDFilteredView is not what is expected \n"
          << ExtractStringFromDDD<cms::DDFilteredView>::getString(attribute, &fv) << "\n";
    }
  }

  auto tracker = std::make_unique<GeometricDet>(&fv, GeometricDet::Tracker);
  CmsTrackerBuilder<cms::DDFilteredView> theCmsTrackerBuilder;
  theCmsTrackerBuilder.build(fv, tracker.get(), attribute);

  return tracker;
}
