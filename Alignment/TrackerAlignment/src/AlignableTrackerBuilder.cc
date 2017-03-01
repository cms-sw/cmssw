#include "Alignment/TrackerAlignment/interface/AlignableTrackerBuilder.h"

// Original Author:  Max Stark
//         Created:  Thu, 13 Jan 2016 10:22:57 CET

// geometry
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GluedGeomDet.h"

// alignment
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/CommonAlignment/interface/AlignableDetUnit.h"
#include "Alignment/CommonAlignment/interface/AlignableCompositeBuilder.h"
#include "Alignment/TrackerAlignment/interface/AlignableSiStripDet.h"
#include "Alignment/TrackerAlignment/interface/TrackerAlignableIndexer.h"



//=============================================================================
//===   PUBLIC METHOD IMPLEMENTATION                                        ===
//=============================================================================

//_____________________________________________________________________________
AlignableTrackerBuilder
::AlignableTrackerBuilder(const TrackerGeometry* trackerGeometry,
                          const TrackerTopology* trackerTopology) :
  trackerGeometry(trackerGeometry),
  trackerTopology(trackerTopology),
  alignableObjectId_(trackerGeometry, nullptr, nullptr),
  alignableMap(nullptr),
  trackerAlignmentLevelBuilder_(trackerTopology, trackerGeometry)
{
  std::ostringstream ss;

  switch (alignableObjectId_.geometry()) {
  case AlignableObjectId::Geometry::RunI:    ss << "RunI geometry";    break;
  case AlignableObjectId::Geometry::PhaseI:  ss << "PhaseI geometry";  break;
  case AlignableObjectId::Geometry::PhaseII: ss << "PhaseII geometry"; break;
  default:
    throw cms::Exception("LogicError")
      << "[AlignableTrackerBuilder] unknown version of TrackerGeometry";
  }

  edm::LogInfo("AlignableBuildProcess")
    << "@SUB=AlignableTrackerBuilder::AlignableTrackerBuilder"
    << "GeometryVersion: " << ss.str();
}

//_____________________________________________________________________________
void AlignableTrackerBuilder
::buildAlignables(AlignableTracker* trackerAlignables)
{
  alignableMap = &trackerAlignables->alignableMap;

  // first, build Alignables on module-level (AlignableDetUnits)
  buildAlignableDetUnits();
  // now build the composite Alignables (Ladders, Layers etc.)
  buildAlignableComposites();

  // create pixel-detector
  buildPixelDetector(trackerAlignables);
  // create strip-detector
  buildStripDetector(trackerAlignables);

  // tracker itself is of course also an Alignable
  alignableMap->get("Tracker").push_back(trackerAlignables);
  // id is the id of first component (should be TPBBarrel)
  trackerAlignables->theId = trackerAlignables->components()[0]->id();
}



//=============================================================================
//===   PRIVATE METHOD IMPLEMENTATION                                       ===
//=============================================================================

//_____________________________________________________________________________
void AlignableTrackerBuilder
::buildAlignableDetUnits()
{
  // PixelBarrel
  convertGeomDetsToAlignables(
    trackerGeometry->detsPXB(), alignableObjectId_.idToString(align::TPBModule)
  );

  // PixelEndcap
  convertGeomDetsToAlignables(
    trackerGeometry->detsPXF(), alignableObjectId_.idToString(align::TPEModule)
  );

  // TIB
  convertGeomDetsToAlignables(
    trackerGeometry->detsTIB(), alignableObjectId_.idToString(align::TIBModule)
  );

  // TID
  convertGeomDetsToAlignables(
    trackerGeometry->detsTID(), alignableObjectId_.idToString(align::TIDModule)
  );

  // TOB
  convertGeomDetsToAlignables(
    trackerGeometry->detsTOB(), alignableObjectId_.idToString(align::TOBModule)
  );

  // TEC
  convertGeomDetsToAlignables(
    trackerGeometry->detsTEC(), alignableObjectId_.idToString(align::TECModule)
  );
}

//_____________________________________________________________________________
void AlignableTrackerBuilder
::convertGeomDetsToAlignables(const TrackingGeometry::DetContainer& geomDets,
                              const std::string& moduleName)
{
  numDetUnits = 0;

  auto& alignables = alignableMap->get(moduleName);
  alignables.reserve(geomDets.size());

  // units are added for each moduleName, which are at moduleName + "Unit"
  // in the pixel Module and ModuleUnit are equivalent
  auto & aliUnits = alignableMap->get(moduleName+"Unit");
  aliUnits.reserve(geomDets.size()); // minimal number space needed

  for (auto& geomDet : geomDets) {
    int subdetId = geomDet->geographicalId().subdetId(); //don't check det()==Tracker

    if (subdetId == PixelSubdetector::PixelBarrel ||
        subdetId == PixelSubdetector::PixelEndcap) {
      buildPixelDetectorAlignable(geomDet, subdetId, alignables, aliUnits);

    } else if (subdetId == SiStripDetId::TIB ||
               subdetId == SiStripDetId::TID ||
               subdetId == SiStripDetId::TOB ||
               subdetId == SiStripDetId::TEC) {
      // for strip we create also <TIB/TID/TOB/TEC>ModuleUnit list
      // for 1D components of 2D layers
      buildStripDetectorAlignable(geomDet, subdetId, alignables, aliUnits);

    } else {
      throw cms::Exception("LogicError")
        << "[AlignableTrackerBuilder] GeomDet of unknown subdetector";
    }

    trackerAlignmentLevelBuilder_.addDetUnitInfo(geomDet->geographicalId());
  }

  // JFI: For PXB and PXE we exclusively build AlignableDetUnit, hence
  // alignables.size() and numDetUnits are equal. But for modules in Strip
  // we also create AlignableSiStripDets, which consist of multiple
  // AlignableDetUnits, hence alignables.size() and numDetUnits are not equal.

  edm::LogInfo("AlignableBuildProcess")
    << "@SUB=AlignableTrackerBuilder::convertGeomDetsToAlignables"
    << "converted GeomDets to Alignables for " << moduleName << "\n"
    << "   GeomDets:             " << geomDets.size()         << "\n"
    << "   AlignableDetUnits:    " << numDetUnits;
}

//_____________________________________________________________________________
void AlignableTrackerBuilder
::buildPixelDetectorAlignable(const GeomDet* geomDetUnit, int subdetId,
                              Alignables& aliDets, Alignables& aliDetUnits)
{
  // treat all pixel dets in same way with one AlignableDetUnit
  if (!geomDetUnit->isLeaf()) {
    throw cms::Exception("BadHierarchy")
      << "[AlignableTrackerBuilder] Pixel GeomDet (subdetector " << subdetId
      << ") is not a GeomDetUnit.";
  }

  aliDets.push_back(new AlignableDetUnit(geomDetUnit));
  aliDetUnits.push_back(aliDets.back());
  numDetUnits += 1;
}

//_____________________________________________________________________________
void AlignableTrackerBuilder
::buildStripDetectorAlignable(const GeomDet* geomDet, int subdetId,
                              Alignables& aliDets, Alignables& aliDetUnits)
{
  // In strip we have:
  // 1) 'Pure' 1D-modules like TOB layers 3-6 (not glued): AlignableDetUnit
  // 2) Composite 2D-modules like TOB layers 1&2 (not glued): AlignableDet
  // 3) The two 1D-components of case 2 (glued): AlignableDetUnit that is constructed
  //      inside AlignableDet-constructor of 'mother', only need to add to alignableLists
  const SiStripDetId detId(geomDet->geographicalId());

  // 2D- or 'pure' 1D-module
  if (!detId.glued()) {
    if (geomDet->components().size()) {
      // 2D-module, convert it to GluedGeomDet
      const GluedGeomDet* gluedGeomDet = dynamic_cast<const GluedGeomDet*>(geomDet);
      if (!gluedGeomDet) {
        throw cms::Exception("LogicError")
          << "[AlignableTrackerBuilder] dynamic_cast<const GluedGeomDet*> "
          << "failed.";
      }

      // components (AlignableDetUnits) constructed within
      aliDets.push_back(new AlignableSiStripDet(gluedGeomDet));
      const auto& addAliDetUnits = aliDets.back()->components();
      const auto& nAddedUnits = addAliDetUnits.size();
      // reserve space for the additional units:
      aliDetUnits.reserve(aliDetUnits.size() + nAddedUnits -1);
      aliDetUnits.insert(aliDetUnits.end(), addAliDetUnits.begin(), addAliDetUnits.end());
      numDetUnits += nAddedUnits;

    } else {
      // no components: pure 1D-module
      buildPixelDetectorAlignable(geomDet, subdetId, aliDets, aliDetUnits);
    }
  } // no else: glued components of AlignableDet constructed within
    // AlignableSiStripDet -> AlignableDet, see above
}



//_____________________________________________________________________________
void AlignableTrackerBuilder
::buildAlignableComposites()
{
  unsigned int numCompositeAlignables = 0;

  // tracker levels must be built before the indexer is created in order to pass
  // a valid namespace to the indexer; an exception would be thrown if one tries
  // to get the namespace w/o building the levels
  auto trackerLevels = trackerAlignmentLevelBuilder_.build();
  TrackerAlignableIndexer trackerIndexer{trackerAlignmentLevelBuilder_.trackerNameSpace()};
  AlignableCompositeBuilder compositeBuilder{trackerTopology, trackerGeometry, trackerIndexer};

  for (auto& trackerSubLevels: trackerLevels) {
    // first add all levels of the current subdetector to the builder
    for (auto& level: trackerSubLevels) {
      compositeBuilder.addAlignmentLevel(std::move(level));
    }
    // now build this tracker-level
    numCompositeAlignables += compositeBuilder.buildAll(*alignableMap);
    // finally, reset the builder
    compositeBuilder.clearAlignmentLevels();
  }

  edm::LogInfo("AlignableBuildProcess")
    << "@SUB=AlignableTrackerBuilder::buildAlignableComposites"
    << "AlignableComposites built for Tracker: " << numCompositeAlignables
    << " (note: without Pixel- and Strip-Alignable)";
}

//_____________________________________________________________________________
void AlignableTrackerBuilder
::buildPixelDetector(AlignableTracker* trackerAlignables)
{
  const std::string& pxbName   = alignableObjectId_.idToString(align::TPBBarrel);
  const std::string& pxeName   = alignableObjectId_.idToString(align::TPEEndcap);
  const std::string& pixelName = alignableObjectId_.idToString(align::Pixel);

  auto& pxbAlignables   = alignableMap->find(pxbName);
  auto& pxeAlignables   = alignableMap->find(pxeName);
  auto& pixelAlignables = alignableMap->get (pixelName);

  pixelAlignables.push_back(
    new AlignableComposite(pxbAlignables[0]->id(), align::Pixel, align::RotationType())
  );

  pixelAlignables[0]->addComponent(pxbAlignables[0]);
  pixelAlignables[0]->addComponent(pxeAlignables[0]);
  pixelAlignables[0]->addComponent(pxeAlignables[1]);

  trackerAlignables->addComponent(pixelAlignables[0]);

  edm::LogInfo("AlignableBuildProcess")
    << "@SUB=AlignableTrackerBuilder::buildPixelDetector"
    << "Built " << pixelName << "-detector Alignable, consisting of Alignables"
    << " of " << pxbName << " and " << pxeName;
}

//_____________________________________________________________________________
void AlignableTrackerBuilder
::buildStripDetector(AlignableTracker* trackerAlignables)
{
  const std::string& tibName   = alignableObjectId_.idToString(align::TIBBarrel);
  const std::string& tidName   = alignableObjectId_.idToString(align::TIDEndcap);
  const std::string& tobName   = alignableObjectId_.idToString(align::TOBBarrel);
  const std::string& tecName   = alignableObjectId_.idToString(align::TECEndcap);
  const std::string& stripName = alignableObjectId_.idToString(align::Strip);

  auto& tibAlignables   = alignableMap->find(tibName);
  auto& tidAlignables   = alignableMap->find(tidName);
  auto& tobAlignables   = alignableMap->find(tobName);
  auto& tecAlignables   = alignableMap->find(tecName);
  auto& stripAlignables = alignableMap->get (stripName);

  stripAlignables.push_back(
    new AlignableComposite(tibAlignables[0]->id(), align::Strip, align::RotationType())
  );

  stripAlignables[0]->addComponent(tibAlignables[0]);
  stripAlignables[0]->addComponent(tidAlignables[0]);
  stripAlignables[0]->addComponent(tidAlignables[1]);
  stripAlignables[0]->addComponent(tobAlignables[0]);
  stripAlignables[0]->addComponent(tecAlignables[0]);
  stripAlignables[0]->addComponent(tecAlignables[1]);

  trackerAlignables->addComponent(stripAlignables[0]);

  edm::LogInfo("AlignableBuildProcess")
    << "@SUB=AlignableTrackerBuilder::buildStripDetector"
    << "Built " << stripName << "-detector Alignable, consisting of Alignables"
    << " of " << tibName << ", " << tidName
    << ", " << tobName << " and " << tecName;
}
