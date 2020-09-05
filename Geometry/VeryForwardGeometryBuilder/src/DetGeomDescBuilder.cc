#include "Geometry/VeryForwardGeometryBuilder/interface/DetGeomDescBuilder.h"

#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


/*
 * Generic function to build geo (tree of DetGeomDesc) from old DD compact view.
 */
std::unique_ptr<DetGeomDesc> detgeomdescbuilder::buildDetGeomDescFromCompactView(const DDCompactView& myCompactView) {
// Create DDFilteredView (no filter!!)
DDPassAllFilter filter;
DDFilteredView fv(myCompactView, filter);

// Geo info: sentinel node.
auto geoInfoSentinel = std::make_unique<DetGeomDesc>(fv);

// Construct the tree of children geo info (DetGeomDesc).
detgeomdescbuilder::buildDetGeomDescDescendants(fv, geoInfoSentinel.get());

edm::LogInfo("PPSGeometryESProducer") << "Successfully built geometry.";

return geoInfoSentinel;
}


/*
 * Use in depth-first search recursion.
 * Construct the tree of children geo info (DetGeomDesc) (old DD navigation).
 */
void detgeomdescbuilder::buildDetGeomDescDescendants(DDFilteredView& fv, DetGeomDesc* geoInfoParent) {
// Leaf
if (!fv.firstChild())
  return;

do {
// Create node, and add it to the geoInfoParent's list.
DetGeomDesc* child = new DetGeomDesc(fv);
geoInfoParent->addComponent(child);

// Recursion
buildDetGeomDescDescendants(fv, child);
} while (fv.nextSibling());

fv.parent();
}


/*
 * Generic function to build geo (tree of DetGeomDesc) from DD4hep compact view.
 */
std::unique_ptr<DetGeomDesc> detgeomdescbuilder::buildDetGeomDescFromCompactView(
    const cms::DDCompactView& myCompactView) {
  // create DDFilteredView (no filter!!)
  const cms::DDDetector* mySystem = myCompactView.detector();
  const dd4hep::Volume& worldVolume = mySystem->worldVolume();
  cms::DDFilteredView fv(mySystem, worldVolume);
  if (fv.next(0) == false) {
    edm::LogError("PPSGeometryESProducer") << "Filtered view is empty. Cannot build.";
  }

  const cms::DDSpecParRegistry& allSpecParSections = myCompactView.specpars();
  // Geo info: sentinel node.
  auto geoInfoSentinel = std::make_unique<DetGeomDesc>(fv, allSpecParSections);

  // Construct the tree of children geo info (DetGeomDesc).
  do {
    // Create node, and add it to the geoInfoSentinel's list.
    DetGeomDesc* newGD = new DetGeomDesc(fv, allSpecParSections);
    geoInfoSentinel->addComponent(newGD);
  } while (fv.next(0));

  edm::LogInfo("PPSGeometryESProducer") << "Successfully built geometry, it has "
                                        << (geoInfoSentinel->components()).size() << " DetGeomDesc nodes.";

  return geoInfoSentinel;
}
