#include "Geometry/TrackerNumberingBuilder/plugins/DDDCmsTrackerContruction.h"

#include <deque>
#include <fstream>
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

#define DEBUG false

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

  edm::LogVerbatim("DDDCmsTrackerContruction") << "DDDCmsTrackerContruction::construct: Call Tracker builder.";
  CmsTrackerBuilder<DDFilteredView> theCmsTrackerBuilder;
  theCmsTrackerBuilder.build(fv, tracker.get(), attribute);

  edm::LogVerbatim("DDDCmsTrackerContruction") << "Assign DetIds";
  CmsTrackerDetIdBuilder theCmsTrackerDetIdBuilder(detidShifts);
  theCmsTrackerDetIdBuilder.buildId(*tracker);

  if (DEBUG) {
    printAllTrackerGeometricDetsBeforeDetIDBuilding(tracker.get());
  }

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
  fv.firstChild();

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

  edm::LogVerbatim("DDDCmsTrackerContruction") << "DDDCmsTrackerContruction::construct: Call Tracker builder.";
  CmsTrackerBuilder<cms::DDFilteredView> theCmsTrackerBuilder;
  theCmsTrackerBuilder.build(fv, tracker.get(), attribute);

  edm::LogVerbatim("DDDCmsTrackerContruction") << "Assign DetIds";
  CmsTrackerDetIdBuilder theCmsTrackerDetIdBuilder(detidShifts);
  theCmsTrackerDetIdBuilder.buildId(*tracker);

  if (DEBUG) {
    printAllTrackerGeometricDetsBeforeDetIDBuilding(tracker.get());
  }

  return tracker;
}

/*
 * Print all Tracker GeometricDets, before DetIds building process.
 * The tree is already fully constructed from XMLs, 
 * and all GeometricDets are sorted according to their geometric position.
 * This allows a convenient debugging, as the DetIds will be later assigned according to this information.
 */
void DDDCmsTrackerContruction::printAllTrackerGeometricDetsBeforeDetIDBuilding(const GeometricDet* tracker) {
  std::ofstream outputFile("All_Tracker_GeometricDets_before_DetId_building.log", std::ios::out);

  // Tree navigation: queue for BFS (we want to see same hierarchy level together).
  // (for DFS, would just use a stack instead).
  std::deque<const GeometricDet*> queue;
  queue.emplace_back(tracker);

  while (!queue.empty()) {
    const GeometricDet* myDet = queue.front();
    queue.pop_front();

    for (auto& child : myDet->components()) {
      queue.emplace_back(child);
    }

    outputFile << " " << std::endl;
    outputFile << " " << std::endl;
    outputFile << "............................." << std::endl;
    outputFile << "myDet->geographicalID() = " << myDet->geographicalId() << std::endl;
    outputFile << "myDet->name() = " << myDet->name() << std::endl;
    outputFile << "myDet->module->type() = " << std::fixed << std::setprecision(7) << myDet->type() << std::endl;
    outputFile << "myDet->module->translation() = " << std::fixed << std::setprecision(7) << myDet->translation()
               << std::endl;
    outputFile << "myDet->module->rho() = " << std::fixed << std::setprecision(7) << myDet->rho() << std::endl;
    if (fabs(myDet->rho()) > 0.00001) {
      outputFile << "myDet->module->phi() = " << std::fixed << std::setprecision(7) << myDet->phi() << std::endl;
    }
    outputFile << "myDet->module->rotation() = " << std::fixed << std::setprecision(7) << myDet->rotation()
               << std::endl;
    outputFile << "myDet->module->shape() = " << std::fixed << std::setprecision(7) << myDet->shape() << std::endl;
    if (myDet->shape_dd4hep() == cms::DDSolidShape::ddbox || myDet->shape_dd4hep() == cms::DDSolidShape::ddtrap ||
        myDet->shape_dd4hep() == cms::DDSolidShape::ddtubs) {
      outputFile << "myDet->params() = " << std::fixed << std::setprecision(7);
      for (const auto& para : myDet->params()) {
        outputFile << para << "  ";
      }
      outputFile << " " << std::endl;
    }
    outputFile << "myDet->radLength() = " << myDet->radLength() << std::endl;
    outputFile << "myDet->xi() = " << myDet->xi() << std::endl;
    outputFile << "myDet->pixROCRows() = " << myDet->pixROCRows() << std::endl;
    outputFile << "myDet->pixROCCols() = " << myDet->pixROCCols() << std::endl;
    outputFile << "myDet->pixROCx() = " << myDet->pixROCx() << std::endl;
    outputFile << "myDet->pixROCy() = " << myDet->pixROCy() << std::endl;
    outputFile << "myDet->stereo() = " << myDet->stereo() << std::endl;
    outputFile << "myDet->isLowerSensor() = " << myDet->isLowerSensor() << std::endl;
    outputFile << "myDet->isUpperSensor() = " << myDet->isUpperSensor() << std::endl;
    outputFile << "myDet->siliconAPVNum() = " << myDet->siliconAPVNum() << std::endl;
  }
}
