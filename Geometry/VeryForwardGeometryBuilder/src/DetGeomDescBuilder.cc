#include "Geometry/VeryForwardGeometryBuilder/interface/DetGeomDescBuilder.h"

#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


/*
 * Generic function to build geo (tree of DetGeomDesc) from old DD compact view.
 */
std::unique_ptr<DetGeomDesc> detgeomdescbuilder::buildDetGeomDescFromCompactView(const DDCompactView& myCompactView) {
  // Create DDFilteredView (no filter!!)
  DDPassAllFilter filter;
  DDFilteredView fv(myCompactView, filter);

  // Geo info: root node.
  auto geoInfoRoot = std::make_unique<DetGeomDesc>(fv);

  // Construct the tree of children geo info (DetGeomDesc).
  detgeomdescbuilder::buildDetGeomDescDescendants(fv, geoInfoRoot.get());

  edm::LogInfo("PPSGeometryESProducer") << "Successfully built geometry.";

  return geoInfoRoot;
}

/*
 * Depth-first search recursion.
 * Construct the tree of children geo info (DetGeomDesc) (old DD).
 */
void detgeomdescbuilder::buildDetGeomDescDescendants(DDFilteredView& fv, DetGeomDesc* geoInfo) {
  // Leaf
  if (!fv.firstChild())
    return;

  do {
    // Create node, and add it to the geoInfoParent's list.
    DetGeomDesc* child = new DetGeomDesc(fv);
    geoInfo->addComponent(child);

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

  // Geo info: root node.
  auto geoInfoRoot = std::make_unique<DetGeomDesc>(fv);

  // Construct the tree of children geo info (DetGeomDesc).
  do {
    // Create node, and add it to the geoInfoRoot's list.
    DetGeomDesc* child = new DetGeomDesc(fv);
    geoInfoRoot->addComponent(child);
  } while (fv.next(0));

  edm::LogInfo("PPSGeometryESProducer") << "Successfully built geometry, it has " << (geoInfoRoot->components()).size()
                                        << " DetGeomDesc nodes.";

  return geoInfoRoot;
}
