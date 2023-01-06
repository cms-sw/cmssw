// Original Author:  Max Stark
//         Created:  Thu, 14 Jan 2016 11:35:07 CET

#include "TrackerGeometryAnalyzer.h"

// for creation of TrackerGeometry
#include "CondFormats/GeometryObjects/interface/PTrackerParameters.h"
#include "Geometry/Records/interface/PTrackerParametersRcd.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"

// for creation of TrackerTopology
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

// tracker-alignables aka AlignableTracker
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/TrackerAlignment/interface/AlignableSiStripDet.h"

//=============================================================================
//===   PUBLIC METHOD IMPLEMENTATION                                        ===
//=============================================================================

//_____________________________________________________________________________
TrackerGeometryAnalyzer ::TrackerGeometryAnalyzer(const edm::ParameterSet& config)
    : tTopoToken_(esConsumes()),
      geomDetToken_(esConsumes()),
      ptpToken_(esConsumes()),
      analyzeAlignables_(config.getParameter<bool>("analyzeAlignables")),
      printTrackerStructure_(config.getParameter<bool>("printTrackerStructure")),
      maxPrintDepth_(config.getParameter<int>("maxPrintDepth")),
      analyzeGeometry_(config.getParameter<bool>("analyzeGeometry")),
      analyzePXB_(config.getParameter<bool>("analyzePXB")),
      analyzePXE_(config.getParameter<bool>("analyzePXE")),
      analyzeTIB_(config.getParameter<bool>("analyzeTIB")),
      analyzeTID_(config.getParameter<bool>("analyzeTID")),
      analyzeTOB_(config.getParameter<bool>("analyzeTOB")),
      analyzeTEC_(config.getParameter<bool>("analyzeTEC")),

      trackerTopology(0),
      trackerGeometry(0),
      // will be reset once the geometry is known:
      alignableObjectId_{AlignableObjectId::Geometry::General} {}

//_____________________________________________________________________________
void TrackerGeometryAnalyzer ::beginRun(const edm::Run& /* run */, const edm::EventSetup& setup) {
  edm::LogInfo("TrackerGeometryAnalyzer") << "@SUB=TrackerGeometryAnalyzer::beginRun"
                                          << "Initializing TrackerGeometryAnalyzer";

  setTrackerTopology(setup);
  setTrackerGeometry(setup);

  if (analyzeAlignables_)
    analyzeTrackerAlignables();
  if (analyzeGeometry_)
    analyzeTrackerGeometry();
}

//=============================================================================
//===   PRIVATE METHOD IMPLEMENTATION                                       ===
//=============================================================================

//_____________________________________________________________________________
void TrackerGeometryAnalyzer ::setTrackerTopology(const edm::EventSetup& setup) {
  trackerTopology = &setup.getData(tTopoToken_);
}

//_____________________________________________________________________________
void TrackerGeometryAnalyzer ::setTrackerGeometry(const edm::EventSetup& setup) {
  edm::ESHandle<GeometricDet> geometricDet = setup.getHandle(geomDetToken_);
  edm::ESHandle<PTrackerParameters> trackerParams = setup.getHandle(ptpToken_);

  TrackerGeomBuilderFromGeometricDet trackerGeometryBuilder;
  trackerGeometry = trackerGeometryBuilder.build(&(*geometricDet), *trackerParams, trackerTopology);
  alignableObjectId_ = AlignableObjectId{trackerGeometry, nullptr, nullptr, nullptr};
}

//_____________________________________________________________________________
void TrackerGeometryAnalyzer ::analyzeTrackerAlignables() {
  edm::LogInfo("TrackerGeometryAnalyzer") << "@SUB=TrackerGeometryAnalyzer::analyzeTrackerAlignables"
                                          << "Building and analyzing TrackerAlignables aka AlignableTracker";

  AlignableTracker* trackerAlignables = new AlignableTracker(trackerGeometry, trackerTopology);

  if (trackerAlignables) {
    analyzeAlignableDetUnits(trackerAlignables);
    analyzeCompositeAlignables(trackerAlignables);

    if (printTrackerStructure_) {
      std::ostringstream ss;

      ss << "\n\n===========================================================\n";
      ss << "TrackerAlignable-structure:\n\n";
      printAlignableStructure(trackerAlignables, ss, 0);
      ss << "\n===========================================================\n\n";

      edm::LogInfo("TrackerGeometryAnalyzer") << "@SUB=TrackerGeometryAnalyzer::printAlignableStructure" << ss.str();
    }

  } else {
    edm::LogError("TrackerGeometryAnalyzer") << "@SUB=TrackerGeometryAnalyzer::analyzeTrackerAlignables"
                                             << "Failed to built AlignableTracker";
  }

  delete trackerAlignables;
}

//_____________________________________________________________________________
void TrackerGeometryAnalyzer ::analyzeAlignableDetUnits(Alignable* trackerAlignables) {
  int numPXB = 0, numPXE = 0, numTIB = 0, numTID = 0, numTOB = 0, numTEC = 0;

  auto allAlignableDetUnits = trackerAlignables->deepComponents();

  for (auto* alignable : allAlignableDetUnits) {
    int num = alignable->deepComponents().size();

    int subdetId = alignable->geomDetId().subdetId();
    switch (subdetId) {
      case PixelSubdetector::PixelBarrel:
        numPXB += num;
        break;
      case PixelSubdetector::PixelEndcap:
        numPXE += num;
        break;
      case StripSubdetector::TIB:
        numTIB += num;
        break;
      case StripSubdetector::TID:
        numTID += num;
        break;
      case StripSubdetector::TOB:
        numTOB += num;
        break;
      case StripSubdetector::TEC:
        numTEC += num;
        break;
    }
  }

  int numDetUnits = numPXB + numPXE + numTIB + numTID + numTOB + numTEC;

  edm::LogInfo("TrackerGeometryAnalyzer") << "@SUB=TrackerGeometryAnalyzer::analyzeAlignableDetUnits"
                                          << "AlignableDetUnits: " << allAlignableDetUnits.size() << "\n"
                                          << "   PXB AlignableDetUnits: " << numPXB << "\n"
                                          << "   PXE AlignableDetUnits: " << numPXE << "\n"
                                          << "   TIB AlignableDetUnits: " << numTIB << "\n"
                                          << "   TID AlignableDetUnits: " << numTID << "\n"
                                          << "   TID AlignableDetUnits: " << numTOB << "\n"
                                          << "   TEC AlignableDetUnits: " << numTEC << "\n"
                                          << "                        ========"
                                          << "\n"
                                          << "                          " << numDetUnits;
}

//_____________________________________________________________________________
void TrackerGeometryAnalyzer ::analyzeCompositeAlignables(Alignable* trackerAlignables) {
  int numPXBComposites = countCompositeAlignables(
                             // PixelBarrel Alignables
                             trackerAlignables->components()[0]->components()[0]) +
                         1;  // + 1 for the Barrel itself
  int numPXEComposites = countCompositeAlignables(
                             // PixelEndcap+ Alignables
                             trackerAlignables->components()[0]->components()[1]) +
                         1 +
                         countCompositeAlignables(
                             // PixelEndcap- Alignables
                             trackerAlignables->components()[0]->components()[2]) +
                         1;
  int numTIBComposites = countCompositeAlignables(
                             // TIB Alignables
                             trackerAlignables->components()[1]->components()[0]) +
                         1;
  int numTIDComposites = countCompositeAlignables(
                             // TID+ Alignables
                             trackerAlignables->components()[1]->components()[1]) +
                         1 +
                         countCompositeAlignables(
                             // TID- Alignables
                             trackerAlignables->components()[1]->components()[2]) +
                         1;
  int numTOBComposites = countCompositeAlignables(
                             // TOB Alignables
                             trackerAlignables->components()[1]->components()[3]) +
                         1;
  int numTECComposites = countCompositeAlignables(
                             // TEC+ Alignables
                             trackerAlignables->components()[1]->components()[4]) +
                         1 +
                         countCompositeAlignables(
                             // TEC- Alignables
                             trackerAlignables->components()[1]->components()[5]) +
                         1;

  int numComposites =
      numPXBComposites + numPXEComposites + numTIBComposites + numTIDComposites + numTOBComposites + numTECComposites;

  edm::LogInfo("TrackerGeometryAnalyzer")
      << "@SUB=TrackerGeometryAnalyzer::analyzeCompositeAlignables"
      << "AlignableComposites: " << countCompositeAlignables(trackerAlignables) << "\n"
      << "   CompositeAlignable in PXB: " << numPXBComposites << "\n"
      << "   CompositeAlignable in PXE: " << numPXEComposites << "\n"
      << "   CompositeAlignable in TIB: " << numTIBComposites << "\n"
      << "   CompositeAlignable in TID: " << numTIDComposites << "\n"
      << "   CompositeAlignable in TOB: " << numTOBComposites << "\n"
      << "   CompositeAlignable in TEC: " << numTECComposites << "\n"
      << "                            ========"
      << "\n"
      << "                              " << numComposites << "\n"
      << "     + 1 Alignable for Pixel: 1"
      << "\n"
      << "     + 1 Alignable for Strip: 1"
      << "\n"
      << "                            ========"
      << "\n"
      << "                              " << numComposites + 2;
}

//_____________________________________________________________________________
int TrackerGeometryAnalyzer ::countCompositeAlignables(Alignable* compositeAlignable) {
  int num = 0;

  for (auto* alignable : compositeAlignable->components()) {
    // We don't count AlignableSiStripDets as CompositeAlignables, but they
    // consist of (usually 2) multiple AlignableDetUnits, hence alignable->
    // components().size() is > 0. To check if we have an AlignableSiStripDet
    // we simple cast it and see if the cast fails or not.
    AlignableSiStripDet* isSiStripDet = dynamic_cast<AlignableSiStripDet*>(alignable);
    if (!isSiStripDet) {
      if (alignable->components().size()) {
        ++num;
        num += countCompositeAlignables(alignable);
      }
    }
  }

  return num;
}

//_____________________________________________________________________________
void TrackerGeometryAnalyzer ::printAlignableStructure(Alignable* compositeAlignable,
                                                       std::ostringstream& ss,
                                                       int indent) {
  if (indent == maxPrintDepth_)
    return;

  for (auto* alignable : compositeAlignable->components()) {
    if (alignable->components().size()) {
      for (int i = 0; i < (3 * indent); ++i)
        ss << " ";

      auto type = alignableObjectId_.idToString(alignable->alignableObjectId());
      ss << type;

      int len = (6 * maxPrintDepth_) - (3 * indent) - strlen(type);
      for (int i = 0; i < len; ++i)
        ss << " ";

      ss << " >> consists of " << alignable->components().size() << " Alignable(s)"
         << "\n";

      printAlignableStructure(alignable, ss, indent + 1);
    }
  }
}

//_____________________________________________________________________________
void TrackerGeometryAnalyzer ::analyzeTrackerGeometry() {
  edm::LogInfo("TrackerGeometryAnalyzer") << "@SUB=TrackerGeometryAnalyzer::analyzeTrackerGeometry"
                                          << "Analyzing TrackerGeometry";
  std::ostringstream ss;

  analyzeTrackerGeometryVersion(ss);

  for (auto& det : trackerGeometry->dets()) {
    auto detId = det->geographicalId();

    if (detId.det() == DetId::Tracker) {
      switch (detId.subdetId()) {
        case PixelSubdetector::PixelBarrel:
          if (analyzePXB_)
            analyzePXBDetUnit(detId, ss);
          break;
        case PixelSubdetector::PixelEndcap:
          if (analyzePXE_)
            analyzePXEDetUnit(detId, ss);
          break;
        case StripSubdetector::TIB:
          if (analyzeTIB_)
            analyzeTIBDetUnit(detId, ss);
          break;
        case StripSubdetector::TID:
          if (analyzeTID_)
            analyzeTIDDetUnit(detId, ss);
          break;
        case StripSubdetector::TOB:
          if (analyzeTOB_)
            analyzeTOBDetUnit(detId, ss);
          break;
        case StripSubdetector::TEC:
          if (analyzeTEC_)
            analyzeTECDetUnit(detId, ss);
          break;
      }
    }
  }

  edm::LogInfo("TrackerGeometryAnalyzer") << "@SUB=TrackerGeometryAnalyzer::analyzeTrackerGeometry" << ss.str();

  if (analyzePXB_)
    analyzePXB();
  if (analyzePXE_)
    analyzePXE();
  if (analyzeTIB_)
    analyzeTIB();
  if (analyzeTID_)
    analyzeTID();
  if (analyzeTOB_)
    analyzeTOB();
  if (analyzeTEC_)
    analyzeTEC();
}

//_____________________________________________________________________________
void TrackerGeometryAnalyzer ::analyzeTrackerGeometryVersion(std::ostringstream& ss) {
  // PixelBarrel
  if (trackerGeometry->isThere(GeomDetEnumerators::PixelBarrel)) {
    ss << "(PXB) PixelBarrel geometry from Run I"
       << "\n";
  } else if (trackerGeometry->isThere(GeomDetEnumerators::P1PXB)) {
    ss << "(PXB) P1PXB geometry from Phase-I "
       << "\n";
  } else if (trackerGeometry->isThere(GeomDetEnumerators::P2PXB)) {
    ss << "(PXB) P2PXB geometry from Phase-II"
       << "\n";
  } else {
    ss << "(PXB) Geometry is unknown"
       << "\n";
  }

  // PixelEndcap
  if (trackerGeometry->isThere(GeomDetEnumerators::PixelEndcap)) {
    ss << "(PXE) PixelEndcap geometry from Run I"
       << "\n";
  } else if (trackerGeometry->isThere(GeomDetEnumerators::P1PXEC)) {
    ss << "(PXE) P1PXEC geometry from Phase-I"
       << "\n";
  } else if (trackerGeometry->isThere(GeomDetEnumerators::P2PXEC)) {
    ss << "(PXE) P2PXEC geometry from Phase-II"
       << "\n";
  } else {
    ss << "(PXE) Geometry is unknown"
       << "\n";
  }

  // TIB
  if (trackerGeometry->isThere(GeomDetEnumerators::TIB)) {
    ss << "(TIB) TIB geometry from Run I and Phase-I"
       << "\n";
  } else if (trackerGeometry->isThere(GeomDetEnumerators::invalidDet)) {
    ss << "(TIB) No TIB geometry since Phase-II"
       << "\n";
  } else {
    ss << "(TIB) Geometry is unknown"
       << "\n";
  }

  // TID
  if (trackerGeometry->isThere(GeomDetEnumerators::TID)) {
    ss << "(TID) TID geometry from Run I and Phase-I"
       << "\n";
  } else if (trackerGeometry->isThere(GeomDetEnumerators::P2OTEC)) {
    ss << "(TID) P2OTEC geometry from Phase-II"
       << "\n";
  } else {
    ss << "(TID) Geometry is unknown"
       << "\n";
  }

  // TOB
  if (trackerGeometry->isThere(GeomDetEnumerators::TOB)) {
    ss << "(TOB) TOB geometry from Run I and Phase-I"
       << "\n";
  } else if (trackerGeometry->isThere(GeomDetEnumerators::P2OTB)) {
    ss << "(TOB) P2OTB geometry from Phase-II"
       << "\n";
  } else {
    ss << "(TOB) Geometry is unknown"
       << "\n";
  }

  // TEC
  if (trackerGeometry->isThere(GeomDetEnumerators::TEC)) {
    ss << "(TEC) TEC geometry from Run I and Phase-I"
       << "\n";
  } else if (trackerGeometry->isThere(GeomDetEnumerators::invalidDet)) {
    ss << "(TEC) No TEC geometry since Phase-II"
       << "\n";
  } else {
    ss << "(TEC) Geometry is unknown"
       << "\n";
  }
}

//_____________________________________________________________________________
void TrackerGeometryAnalyzer ::analyzePXBDetUnit(DetId& detId, std::ostringstream& ss) {
  auto layerID = trackerTopology->pxbLayer(detId);
  auto ladderID = trackerTopology->pxbLadder(detId);
  auto moduleID = trackerTopology->module(detId);

  ss << "Topology info for - TPBModule - with DetId " << detId.rawId() << ": "
     << "   layerID: " << layerID << "   ladderID: " << ladderID << "   moduleID: " << moduleID << "\n";

  pxbLayerIDs.insert(layerID);
  pxbLadderIDs.insert(ladderID);
  pxbModuleIDs.insert(moduleID);

  ++numPXBDetUnits;
}

void TrackerGeometryAnalyzer ::analyzePXB() {
  edm::LogInfo("TrackerGeometryAnalyzer") << "@SUB=TrackerGeometryAnalyzer::analyzePXB"
                                          << "   number of PXBModules:              " << numPXBDetUnits << "\n"
                                          << "   max. number of modules per ladder: " << pxbModuleIDs.size() << "\n"
                                          << "   max. number of ladders per layer:  " << pxbLadderIDs.size() << "\n"
                                          << "   max. number of layers  in PXB:     " << pxbLayerIDs.size();

  std::vector<std::vector<int> > numPXBModules(pxbLayerIDs.size(), std::vector<int>(pxbLadderIDs.size(), 0));

  for (auto& det : trackerGeometry->dets()) {
    auto detId = det->geographicalId();

    if (detId.det() == DetId::Tracker) {
      if (detId.subdetId() == PixelSubdetector::PixelBarrel) {
        auto layerID = trackerTopology->pxbLayer(detId);
        auto ladderID = trackerTopology->pxbLadder(detId);

        numPXBModules[layerID - 1][ladderID - 1]++;
      }
    }
  }

  std::ostringstream ss;
  ss << "analyzing PixelBarrel modules\n";

  for (unsigned int i = 0; i < pxbLayerIDs.size(); ++i) {
    for (unsigned int j = 0; j < pxbLadderIDs.size(); ++j) {
      if (numPXBModules[i][j] > 0) {
        ss << "   number of modules in layer-" << i + 1 << " ladder-" << j + 1 << ": " << numPXBModules[i][j] << "\n";
      }
    }
  }

  edm::LogInfo("TrackerGeometryAnalyzer") << "@SUB=TrackerGeometryAnalyzer::analyzePXB" << ss.str();
}

//_____________________________________________________________________________
void TrackerGeometryAnalyzer ::analyzePXEDetUnit(DetId& detId, std::ostringstream& ss) {
  auto sideID = trackerTopology->pxfSide(detId);
  auto diskID = trackerTopology->pxfDisk(detId);
  auto bladeID = trackerTopology->pxfBlade(detId);
  auto panelID = trackerTopology->pxfPanel(detId);
  auto moduleID = trackerTopology->module(detId);

  ss << "Topology info for - TPEModule - with DetId " << detId.rawId() << ": "
     << "   sideID: " << sideID << "   diskID: " << diskID << "   bladeID: " << bladeID << "   panelID: " << panelID
     << "   moduleID: " << moduleID << "\n";

  pxeSideIDs.insert(sideID);
  pxeDiskIDs.insert(diskID);
  pxeBladeIDs.insert(bladeID);
  pxePanelIDs.insert(panelID);
  pxeModuleIDs.insert(moduleID);

  numPXEDetUnits++;
}

//_____________________________________________________________________________
void TrackerGeometryAnalyzer ::analyzePXE() {
  edm::LogInfo("TrackerGeometryAnalyzer") << "@SUB=TrackerGeometryAnalyzer::analyzePXE"
                                          << "   number of PXEModules:             " << numPXEDetUnits << "\n"
                                          << "   max. number of modules per panel: " << pxeModuleIDs.size() << "\n"
                                          << "   max. number of panels  per blade: " << pxePanelIDs.size() << "\n"
                                          << "   max. number of blades  per disk:  " << pxeBladeIDs.size() << "\n"
                                          << "   max. number of disks   per side:  " << pxeDiskIDs.size() << "\n"
                                          << "   max. number of sides in PXE:      " << pxeSideIDs.size();

  std::vector<std::vector<std::vector<std::vector<int> > > > numAlignablesPXE(
      pxeSideIDs.size(),
      std::vector<std::vector<std::vector<int> > >(
          pxeDiskIDs.size(),
          std::vector<std::vector<int> >(
              pxeBladeIDs.size(), std::vector<int>(pxePanelIDs.size(), 0)  // initialize everything with 0
              )));

  for (auto& det : trackerGeometry->dets()) {
    auto detId = det->geographicalId();

    if (detId.det() == DetId::Tracker) {
      if (detId.subdetId() == PixelSubdetector::PixelEndcap) {
        auto sideID = trackerTopology->pxfSide(detId);
        auto diskID = trackerTopology->pxfDisk(detId);
        auto bladeID = trackerTopology->pxfBlade(detId);
        auto panelID = trackerTopology->pxfPanel(detId);
        //auto moduleID = trackerTopology->module(detId);

        numAlignablesPXE[sideID - 1][diskID - 1][bladeID - 1][panelID - 1]++;
      }
    }
  }

  std::ostringstream ss;
  ss << "analyzing PixelEndcap modules\n";

  for (unsigned int i = 0; i < pxeSideIDs.size(); ++i) {
    for (unsigned int j = 0; j < pxeDiskIDs.size(); ++j) {
      for (unsigned int k = 0; k < pxeBladeIDs.size(); ++k) {
        for (unsigned int l = 0; l < pxePanelIDs.size(); ++l) {
          if (numAlignablesPXE[i][j][k][l] > 0) {
            ss << "   number of modules in side-" << i + 1 << " disk-" << j + 1 << " blade-" << k + 1 << " panel-"
               << l + 1 << ": " << numAlignablesPXE[i][j][k][l] << "\n";
          }
        }
      }
    }
  }

  edm::LogInfo("TrackerGeometryAnalyzer") << "@SUB=TrackerGeometryAnalyzer::analyzePXE" << ss.str();
}

//_____________________________________________________________________________
void TrackerGeometryAnalyzer ::analyzeTIBDetUnit(DetId& detId, std::ostringstream& ss) {
  auto sideID = trackerTopology->tibSide(detId);
  auto layerID = trackerTopology->tibLayer(detId);
  auto stringID = trackerTopology->tibString(detId);
  auto moduleID = trackerTopology->module(detId);

  ss << "Topology info for - TIBModule - with DetId " << detId.rawId() << ": "
     << "   sideID: " << sideID << "   layerID: " << layerID << "   stringID: " << stringID
     << "   moduleID: " << moduleID << "\n";

  tibSideIDs.insert(sideID);
  tibLayerIDs.insert(layerID);
  tibStringIDs.insert(stringID);
  tibModuleIDs.insert(moduleID);

  ++numTIBDetUnits;
}

//_____________________________________________________________________________
void TrackerGeometryAnalyzer ::analyzeTIB() {
  // TODO: not yet implemented
}

//_____________________________________________________________________________
void TrackerGeometryAnalyzer ::analyzeTIDDetUnit(DetId& detId, std::ostringstream& ss) {
  auto sideID = trackerTopology->tidSide(detId);
  auto wheelID = trackerTopology->tidWheel(detId);
  auto ringID = trackerTopology->tidRing(detId);
  auto moduleID = trackerTopology->module(detId);

  ss << "Topology info for - TIDModule - with DetId " << detId.rawId() << ": "
     << "   sideID: " << sideID << "   wheelID: " << wheelID << "   ringID: " << ringID << "   moduleID: " << moduleID
     << "\n";

  tidSideIDs.insert(sideID);
  tidWheelIDs.insert(wheelID);
  tidRingIDs.insert(ringID);
  tidModuleIDs.insert(moduleID);

  ++numTIDDetUnits;
}

//_____________________________________________________________________________
void TrackerGeometryAnalyzer ::analyzeTID() {
  // TODO: not yet implemented
}

//_____________________________________________________________________________
void TrackerGeometryAnalyzer ::analyzeTOBDetUnit(DetId& detId, std::ostringstream& ss) {
  auto layerID = trackerTopology->tobLayer(detId);
  auto sideID = trackerTopology->tobSide(detId);
  auto rodID = trackerTopology->tobRod(detId);
  auto moduleID = trackerTopology->module(detId);

  ss << "Topology info for - TOBModule - with DetId " << detId.rawId() << ": "
     << "   layerID: " << layerID << "   sideID: " << sideID << "   rodID: " << rodID << "   moduleID: " << moduleID
     << "\n";

  tobLayerIDs.insert(layerID);
  tobSideIDs.insert(sideID);
  tobRodIDs.insert(rodID);
  tobModuleIDs.insert(moduleID);

  ++numTOBDetUnits;
}

//_____________________________________________________________________________
void TrackerGeometryAnalyzer ::analyzeTOB() {
  // TODO: not yet implemented
}

//_____________________________________________________________________________
void TrackerGeometryAnalyzer ::analyzeTECDetUnit(DetId& detId, std::ostringstream& ss) {
  auto sideID = trackerTopology->tecSide(detId);
  auto wheelID = trackerTopology->tecWheel(detId);
  auto petalID = trackerTopology->tecPetalNumber(detId);
  auto ringID = trackerTopology->tecRing(detId);
  auto moduleID = trackerTopology->module(detId);

  ss << "Topology info for - TECModule - with DetId " << detId.rawId() << ": "
     << "   sideID: " << sideID << "   wheelID: " << wheelID << "   petalID: " << petalID << "   ringID: " << ringID
     << "   moduleID: " << moduleID << "\n";

  tecSideIDs.insert(sideID);
  tecWheelIDs.insert(wheelID);
  tecPetalIDs.insert(petalID);
  tecRingIDs.insert(ringID);
  tecModuleIDs.insert(moduleID);

  ++numTECDetUnits;
}

//_____________________________________________________________________________
void TrackerGeometryAnalyzer ::analyzeTEC() {
  // TODO: not yet implemented
}
