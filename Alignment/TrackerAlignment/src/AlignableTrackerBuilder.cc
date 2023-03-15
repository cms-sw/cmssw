#include "Alignment/TrackerAlignment/interface/AlignableTrackerBuilder.h"

// Original Author:  Max Stark
//         Created:  Thu, 13 Jan 2016 10:22:57 CET

// geometry
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GluedGeomDet.h"
#include "Geometry/CommonTopologies/interface/StackGeomDet.h"

// alignment
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/CommonAlignment/interface/AlignableDetUnit.h"
#include "Alignment/CommonAlignment/interface/AlignableCompositeBuilder.h"
#include "Alignment/TrackerAlignment/interface/AlignableSiStripDet.h"
#include "Alignment/TrackerAlignment/interface/AlignableStackDet.h"
#include "Alignment/TrackerAlignment/interface/TrackerAlignableIndexer.h"

#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"

//=============================================================================
//===   PUBLIC METHOD IMPLEMENTATION                                        ===
//=============================================================================

//_____________________________________________________________________________
AlignableTrackerBuilder ::AlignableTrackerBuilder(const TrackerGeometry* trackerGeometry,
                                                  const TrackerTopology* trackerTopology)
    : trackerGeometry_(trackerGeometry),
      trackerTopology_(trackerTopology),
      alignableObjectId_(trackerGeometry, nullptr, nullptr, nullptr),
      alignableMap_(nullptr),
      trackerAlignmentLevelBuilder_(trackerTopology, trackerGeometry) {
  std::ostringstream ss;

  switch (alignableObjectId_.geometry()) {
    case AlignableObjectId::Geometry::RunI:
      ss << "RunI geometry";
      break;
    case AlignableObjectId::Geometry::PhaseI:
      ss << "PhaseI geometry";
      break;
    case AlignableObjectId::Geometry::PhaseII:
      ss << "PhaseII geometry";
      break;
    default:
      throw cms::Exception("LogicError") << "[AlignableTrackerBuilder] unknown version of TrackerGeometry";
  }

  edm::LogInfo("AlignableBuildProcess") << "@SUB=AlignableTrackerBuilder::AlignableTrackerBuilder"
                                        << "GeometryVersion: " << ss.str();
}

//_____________________________________________________________________________
void AlignableTrackerBuilder ::buildAlignables(AlignableTracker* trackerAlignables, bool update) {
  alignableMap_ = &trackerAlignables->alignableMap_;

  // first, build Alignables on module-level (AlignableDetUnits)
  buildAlignableDetUnits(update);

  // now build the composite Alignables (Ladders, Layers etc.)
  buildAlignableComposites(update);

  if (update)
    return;  // everything else not needed for the update

  // create pixel-detector
  buildPixelDetector(trackerAlignables);

  // for the Outer Tracker, decide which geometry we are addressing
  if (alignableObjectId_.geometry() < AlignableObjectId::Geometry::PhaseII) {
    // create strip-detector
    buildStripDetector(trackerAlignables);
  } else {
    // create Phase2 Outer Tracker-detector
    buildOuterTrackerDetector(trackerAlignables);
  }

  // tracker itself is of course also an Alignable
  alignableMap_->get("Tracker").push_back(trackerAlignables);
  // id is the id of first component (should be TPBBarrel)
  trackerAlignables->theId = trackerAlignables->components()[0]->id();
}

//=============================================================================
//===   PRIVATE METHOD IMPLEMENTATION                                       ===
//=============================================================================

//_____________________________________________________________________________
void AlignableTrackerBuilder ::buildAlignableDetUnits(bool update) {
  // PixelBarrel
  convertGeomDetsToAlignables(trackerGeometry_->detsPXB(), alignableObjectId_.idToString(align::TPBModule), update);

  // PixelEndcap
  convertGeomDetsToAlignables(trackerGeometry_->detsPXF(), alignableObjectId_.idToString(align::TPEModule), update);

  // TIB
  convertGeomDetsToAlignables(trackerGeometry_->detsTIB(), alignableObjectId_.idToString(align::TIBModule), update);

  // TID
  convertGeomDetsToAlignables(trackerGeometry_->detsTID(), alignableObjectId_.idToString(align::TIDModule), update);

  // TOB
  convertGeomDetsToAlignables(trackerGeometry_->detsTOB(), alignableObjectId_.idToString(align::TOBModule), update);

  // TEC
  convertGeomDetsToAlignables(trackerGeometry_->detsTEC(), alignableObjectId_.idToString(align::TECModule), update);
}

//_____________________________________________________________________________
void AlignableTrackerBuilder ::convertGeomDetsToAlignables(const TrackingGeometry::DetContainer& geomDets,
                                                           const std::string& moduleName,
                                                           bool update) {
  numDetUnits = 0;

  auto& alignables = alignableMap_->get(moduleName);
  if (!update)
    alignables.reserve(geomDets.size());

  // units are added for each moduleName, which are at moduleName + "Unit"
  // in the pixel Module and ModuleUnit are equivalent
  auto& aliUnits = alignableMap_->get(moduleName + "Unit");
  if (!update)
    aliUnits.reserve(geomDets.size());  // minimal number space needed

  for (auto& geomDet : geomDets) {
    int subdetId = geomDet->geographicalId().subdetId();  //don't check det()==Tracker

    if (subdetId == PixelSubdetector::PixelBarrel || subdetId == PixelSubdetector::PixelEndcap) {
      buildPixelDetectorAlignable(geomDet, subdetId, alignables, aliUnits, update);

    } else if (subdetId == SiStripDetId::TIB || subdetId == SiStripDetId::TID || subdetId == SiStripDetId::TOB ||
               subdetId == SiStripDetId::TEC) {
      // for strip we create also <TIB/TID/TOB/TEC>ModuleUnit list
      // for 1D components of 2D layers

      if (alignableObjectId_.geometry() < AlignableObjectId::Geometry::PhaseII) {
        buildStripDetectorAlignable(geomDet, subdetId, alignables, aliUnits, update);
      } else {
        buildOuterTrackerDetectorAlignable(geomDet, subdetId, alignables, aliUnits, update);
      }

    } else {
      throw cms::Exception("LogicError") << "[AlignableTrackerBuilder] GeomDet of unknown subdetector";
    }

    trackerAlignmentLevelBuilder_.addDetUnitInfo(geomDet->geographicalId());
  }

  // JFI: For PXB and PXE we exclusively build AlignableDetUnit, hence
  // alignables.size() and numDetUnits are equal. But for modules in Strip
  // we also create AlignableSiStripDets, which consist of multiple
  // AlignableDetUnits, hence alignables.size() and numDetUnits are not equal.

  edm::LogInfo("AlignableBuildProcess") << "@SUB=AlignableTrackerBuilder::convertGeomDetsToAlignables"
                                        << "converted GeomDets to Alignables for " << moduleName << "\n"
                                        << "   GeomDets:             " << geomDets.size() << "\n"
                                        << "   AlignableDetUnits:    " << numDetUnits;
}

//_____________________________________________________________________________
void AlignableTrackerBuilder ::buildPixelDetectorAlignable(
    const GeomDet* geomDetUnit, int subdetId, Alignables& aliDets, Alignables& aliDetUnits, bool update) {
  // treat all pixel dets in same way with one AlignableDetUnit
  if (!geomDetUnit->isLeaf()) {
    throw cms::Exception("BadHierarchy") << "[AlignableTrackerBuilder] Pixel GeomDet (subdetector " << subdetId
                                         << ") is not a GeomDetUnit.";
  }

  if (update) {
    auto ali = std::find_if(aliDets.cbegin(), aliDets.cend(), [&geomDetUnit](const auto& i) {
      return i->id() == geomDetUnit->geographicalId().rawId();
    });
    if (ali != aliDets.end()) {
      // add dynamic cast here to get AlignableDetUnit!
      auto aliDetUnit = dynamic_cast<AlignableDetUnit*>(*ali);
      if (aliDetUnit) {
        aliDetUnit->update(geomDetUnit);
      } else {
        throw cms::Exception("LogicError") << "[AlignableTrackerBuilder::buildPixelDetectorAlignable] "
                                           << "cast to 'AlignableDetUnit*' failed while it should not\n";
      }
    } else {
      throw cms::Exception("GeometryMismatch")
          << "[AlignableTrackerBuilder::buildPixelDetectorAlignable] "
          << "GeomDet with DetId " << geomDetUnit->geographicalId().rawId() << " not found in current geometry.\n";
    }
  } else {
    aliDets.push_back(new AlignableDetUnit(geomDetUnit));
    aliDetUnits.push_back(aliDets.back());
  }
  numDetUnits += 1;
}

//_____________________________________________________________________________
void AlignableTrackerBuilder ::buildStripDetectorAlignable(
    const GeomDet* geomDet, int subdetId, Alignables& aliDets, Alignables& aliDetUnits, bool update) {
  // In strip we have:
  // 1) 'Pure' 1D-modules like TOB layers 3-6 (not glued): AlignableDetUnit
  // 2) Composite 2D-modules like TOB layers 1&2 (not glued): AlignableDet
  // 3) The two 1D-components of case 2 (glued): AlignableDetUnit that is constructed
  //      inside AlignableDet-constructor of 'mother', only need to add to alignableLists
  const SiStripDetId detId(geomDet->geographicalId());

  // 2D- or 'pure' 1D-module
  if (!detId.glued()) {
    if (!geomDet->components().empty()) {
      // 2D-module, convert it to GluedGeomDet
      const GluedGeomDet* gluedGeomDet = dynamic_cast<const GluedGeomDet*>(geomDet);
      if (!gluedGeomDet) {
        throw cms::Exception("LogicError") << "[AlignableTrackerBuilder] dynamic_cast<const GluedGeomDet*> "
                                           << "failed.";
      }

      // components (AlignableDetUnits) constructed within
      if (update) {
        auto ali = std::find_if(aliDets.cbegin(), aliDets.cend(), [&gluedGeomDet](const auto& i) {
          return i->id() == gluedGeomDet->geographicalId().rawId();
        });
        if (ali != aliDets.end()) {
          auto aliSiStripDet = dynamic_cast<AlignableSiStripDet*>(*ali);
          if (aliSiStripDet) {
            aliSiStripDet->update(gluedGeomDet);
          } else {
            throw cms::Exception("LogicError") << "[AlignableTrackerBuilder::buildStripDetectorAlignable] "
                                               << "cast to 'AlignableSiStripDet*' failed while it should not\n";
          }
        } else {
          throw cms::Exception("GeometryMismatch")
              << "[AlignableTrackerBuilder::buildStripDetectorAlignable] "
              << "GeomDet with DetId " << gluedGeomDet->geographicalId().rawId() << " not found in current geometry.\n";
        }
      } else {
        aliDets.push_back(new AlignableSiStripDet(gluedGeomDet));
      }
      const auto& addAliDetUnits = aliDets.back()->components();
      const auto& nAddedUnits = addAliDetUnits.size();

      if (!update) {
        // reserve space for the additional units:
        aliDetUnits.reserve(aliDetUnits.size() + nAddedUnits - 1);
        aliDetUnits.insert(aliDetUnits.end(), addAliDetUnits.begin(), addAliDetUnits.end());
      }
      numDetUnits += nAddedUnits;

    } else {
      // no components: pure 1D-module
      buildPixelDetectorAlignable(geomDet, subdetId, aliDets, aliDetUnits, update);
    }
  }  // no else: glued components of AlignableDet constructed within
     // AlignableSiStripDet -> AlignableDet, see above
}

//_____________________________________________________________________________
void AlignableTrackerBuilder ::buildOuterTrackerDetectorAlignable(
    const GeomDet* geomDet, int subdetId, Alignables& aliDets, Alignables& aliDetUnits, bool update) {
  // hopefully all the geomdets are composite (either PS or SS modules in Ph-2 Outer Tracker)
  if (!geomDet->components().empty()) {
    // 2D-module, convert it to StackGeomDet
    const StackGeomDet* stackGeomDet = dynamic_cast<const StackGeomDet*>(geomDet);
    if (!stackGeomDet) {
      throw cms::Exception("LogicError") << "[AlignableTrackerBuilder] dynamic_cast<const StackGeomDet*> "
                                         << "failed.";
    }

    // components (AlignableDetUnits) constructed within
    if (update) {
      auto ali = std::find_if(aliDets.cbegin(), aliDets.cend(), [&stackGeomDet](const auto& i) {
        return i->id() == stackGeomDet->geographicalId().rawId();
      });
      if (ali != aliDets.end()) {
        auto aliStackDet = dynamic_cast<AlignableStackDet*>(*ali);
        if (aliStackDet) {
          aliStackDet->update(geomDet);
        } else {
          throw cms::Exception("LogicError") << "[AlignableTrackerBuilder::buildOuterTrackerDetectorAlignable] "
                                             << "cast to 'AlignableStackDet*' failed while it should not\n";
        }
      } else {
        throw cms::Exception("GeometryMismatch")
            << "[AlignableTrackerBuilder::buildStripDetectorAlignable] "
            << "GeomDet with DetId " << stackGeomDet->geographicalId().rawId() << " not found in current geometry.\n";
      }
    } else {
      aliDets.push_back(new AlignableStackDet(stackGeomDet));
    }
    const auto& addAliDetUnits = aliDets.back()->components();
    const auto& nAddedUnits = addAliDetUnits.size();

    if (!update) {
      // reserve space for the additional units:
      aliDetUnits.reserve(aliDetUnits.size() + nAddedUnits - 1);
      aliDetUnits.insert(aliDetUnits.end(), addAliDetUnits.begin(), addAliDetUnits.end());
    }
    numDetUnits += nAddedUnits;
  }  // no else: stacked components of AlignableDet constructed within
     // AlignableStackDet -> AlignableDet, see above
}

//_____________________________________________________________________________
void AlignableTrackerBuilder ::buildAlignableComposites(bool update) {
  unsigned int numCompositeAlignables = 0;

  // tracker levels must be built before the indexer is created in order to pass
  // a valid namespace to the indexer; an exception would be thrown if one tries
  // to get the namespace w/o building the levels
  auto trackerLevels = trackerAlignmentLevelBuilder_.build();
  TrackerAlignableIndexer trackerIndexer{trackerAlignmentLevelBuilder_.trackerNameSpace()};
  AlignableCompositeBuilder compositeBuilder{trackerTopology_, trackerGeometry_, trackerIndexer};

  for (auto& trackerSubLevels : trackerLevels) {
    // first add all levels of the current subdetector to the builder
    for (auto& level : trackerSubLevels) {
      compositeBuilder.addAlignmentLevel(std::move(level));
    }
    // now build this tracker-level
    numCompositeAlignables += compositeBuilder.buildAll(*alignableMap_, update);
    // finally, reset the builder
    compositeBuilder.clearAlignmentLevels();
  }

  edm::LogInfo("AlignableBuildProcess") << "@SUB=AlignableTrackerBuilder::buildAlignableComposites"
                                        << "AlignableComposites built for Tracker: " << numCompositeAlignables
                                        << " (note: without Pixel- and Strip-Alignable)";
}

//_____________________________________________________________________________
void AlignableTrackerBuilder ::buildPixelDetector(AlignableTracker* trackerAlignables) {
  const std::string& pxbName = alignableObjectId_.idToString(align::TPBBarrel);
  const std::string& pxeName = alignableObjectId_.idToString(align::TPEEndcap);
  const std::string& pixelName = alignableObjectId_.idToString(align::Pixel);

  auto& pxbAlignables = alignableMap_->find(pxbName);
  auto& pxeAlignables = alignableMap_->find(pxeName);
  auto& pixelAlignables = alignableMap_->get(pixelName);

  pixelAlignables.push_back(new AlignableComposite(pxbAlignables[0]->id(), align::Pixel, align::RotationType()));

  pixelAlignables[0]->addComponent(pxbAlignables[0]);
  pixelAlignables[0]->addComponent(pxeAlignables[0]);
  pixelAlignables[0]->addComponent(pxeAlignables[1]);

  trackerAlignables->addComponent(pixelAlignables[0]);

  edm::LogInfo("AlignableBuildProcess") << "@SUB=AlignableTrackerBuilder::buildPixelDetector"
                                        << "Built " << pixelName << "-detector Alignable, consisting of Alignables"
                                        << " of " << pxbName << " and " << pxeName;
}

//_____________________________________________________________________________
void AlignableTrackerBuilder ::buildStripDetector(AlignableTracker* trackerAlignables) {
  const std::string& tibName = alignableObjectId_.idToString(align::TIBBarrel);
  const std::string& tidName = alignableObjectId_.idToString(align::TIDEndcap);
  const std::string& tobName = alignableObjectId_.idToString(align::TOBBarrel);
  const std::string& tecName = alignableObjectId_.idToString(align::TECEndcap);
  const std::string& stripName = alignableObjectId_.idToString(align::Strip);

  auto& tibAlignables = alignableMap_->find(tibName);
  auto& tidAlignables = alignableMap_->find(tidName);
  auto& tobAlignables = alignableMap_->find(tobName);
  auto& tecAlignables = alignableMap_->find(tecName);
  auto& stripAlignables = alignableMap_->get(stripName);

  stripAlignables.push_back(new AlignableComposite(tibAlignables[0]->id(), align::Strip, align::RotationType()));

  stripAlignables[0]->addComponent(tibAlignables[0]);
  stripAlignables[0]->addComponent(tidAlignables[0]);
  stripAlignables[0]->addComponent(tidAlignables[1]);
  stripAlignables[0]->addComponent(tobAlignables[0]);
  stripAlignables[0]->addComponent(tecAlignables[0]);
  stripAlignables[0]->addComponent(tecAlignables[1]);

  trackerAlignables->addComponent(stripAlignables[0]);

  edm::LogInfo("AlignableBuildProcess") << "@SUB=AlignableTrackerBuilder::buildStripDetector"
                                        << "Built " << stripName << "-detector Alignable, consisting of Alignables"
                                        << " of " << tibName << ", " << tidName << ", " << tobName << " and "
                                        << tecName;
}

//_____________________________________________________________________________
void AlignableTrackerBuilder ::buildOuterTrackerDetector(AlignableTracker* trackerAlignables) {
  const std::string& tidName = alignableObjectId_.idToString(align::TIDEndcap);
  const std::string& tobName = alignableObjectId_.idToString(align::TOBBarrel);
  const std::string& stripName = alignableObjectId_.idToString(align::Strip);

  auto& tidAlignables = alignableMap_->find(tidName);
  auto& tobAlignables = alignableMap_->find(tobName);
  auto& stripAlignables = alignableMap_->get(stripName);

  stripAlignables.push_back(new AlignableComposite(tobAlignables[0]->id(), align::Strip, align::RotationType()));
  stripAlignables[0]->addComponent(tobAlignables[0]);
  stripAlignables[0]->addComponent(tidAlignables[0]);
  stripAlignables[0]->addComponent(tidAlignables[1]);

  trackerAlignables->addComponent(stripAlignables[0]);

  edm::LogInfo("AlignableBuildProcess") << "@SUB=AlignableTrackerBuilder::buildStripDetector"
                                        << "Built " << stripName << "-detector Alignable, consisting of Alignables"
                                        << " of " << tidName << " and " << tobName;
}
