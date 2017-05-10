
#include "Alignment/TrackerAlignment/interface/TrackerAlignmentLevelBuilder.h"

// Original Author:  Max Stark
//         Created:  Wed, 10 Feb 2016 13:48:41 CET

#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"

// these extern defined fields (see files TPBNameSpace.h etc.) hold some
// geometry-dependent values -> they will be set in this class
namespace align
{
  namespace tpb { extern std::vector<unsigned int> lpqc; }
  namespace tpe { extern unsigned int bpqd; }
  namespace tib { extern std::vector<unsigned int> sphs; }
}



//=============================================================================
//===   PUBLIC METHOD IMPLEMENTATION                                        ===
//=============================================================================

//_____________________________________________________________________________
TrackerAlignmentLevelBuilder
::TrackerAlignmentLevelBuilder(const TrackerTopology* trackerTopology) :
  trackerTopology(trackerTopology)
{
}

//_____________________________________________________________________________
TrackerAlignmentLevelBuilder::
~TrackerAlignmentLevelBuilder()
{
  // cleanup; AlignmentLevels were created here, so this level-builder is
  // also responsible for deleting them
  for (auto* subLevels : levels) {
    for (auto* level : *subLevels) {
      delete level;
    }
  }
}

//_____________________________________________________________________________
void TrackerAlignmentLevelBuilder
::addDetUnitInfo(const DetId& detId)
{
  int subdetId = detId.subdetId();

  switch (subdetId) {
    case PixelSubdetector::PixelBarrel: addPXBDetUnitInfo(detId); break;
    case PixelSubdetector::PixelEndcap: addPXEDetUnitInfo(detId); break;
    case StripSubdetector::TIB:         addTIBDetUnitInfo(detId); break;
    case StripSubdetector::TID:         addTIDDetUnitInfo(detId); break;
    case StripSubdetector::TOB:         addTOBDetUnitInfo(detId); break;
    case StripSubdetector::TEC:         addTECDetUnitInfo(detId); break;
  }
}

//_____________________________________________________________________________
std::vector<align::AlignmentLevels*> TrackerAlignmentLevelBuilder
::build()
{
  buildPXBAlignmentLevels();
  buildPXEAlignmentLevels();
  buildTIBAlignmentLevels();
  buildTIDAlignmentLevels();
  buildTOBAlignmentLevels();
  buildTECAlignmentLevels();

  levels.push_back(&pxb);
  levels.push_back(&pxe);
  levels.push_back(&tib);
  levels.push_back(&tid);
  levels.push_back(&tob);
  levels.push_back(&tec);

  return levels;
}



//=============================================================================
//===   PRIVATE METHOD IMPLEMENTATION                                       ===
//=============================================================================

//_____________________________________________________________________________
void TrackerAlignmentLevelBuilder
::addPXBDetUnitInfo(const DetId& detId)
{
  auto layerID  = trackerTopology->pxbLayer(detId);
  auto ladderID = trackerTopology->pxbLadder(detId);
  auto moduleID = trackerTopology->module(detId);

  if (pxbLaddersPerLayer[layerID-1] < ladderID) {
    pxbLaddersPerLayer[layerID-1] = ladderID;
  }

  pxbLayerIDs. insert(layerID);
  pxbLadderIDs.insert(ladderID);
  pxbModuleIDs.insert(moduleID);
}

//_____________________________________________________________________________
void TrackerAlignmentLevelBuilder
::addPXEDetUnitInfo(const DetId& detId)
{
  auto sideID   = trackerTopology->pxfSide(detId);
  auto diskID   = trackerTopology->pxfDisk(detId);
  auto bladeID  = trackerTopology->pxfBlade(detId);
  auto panelID  = trackerTopology->pxfPanel(detId);
  auto moduleID = trackerTopology->module(detId);

  pxeSideIDs.  insert(sideID);
  pxeDiskIDs.  insert(diskID);
  pxeBladeIDs. insert(bladeID);
  pxePanelIDs. insert(panelID);
  pxeModuleIDs.insert(moduleID);
}

//_____________________________________________________________________________
void TrackerAlignmentLevelBuilder
::addTIBDetUnitInfo(const DetId& detId)
{
  auto sideID    = trackerTopology->tibSide(detId);
  auto layerID   = trackerTopology->tibLayer(detId);
  auto layerSide = trackerTopology->tibOrder(detId);
  auto stringID  = trackerTopology->tibString(detId);
  auto moduleID  = trackerTopology->module(detId);

  if (layerSide == 1) {
    if (tidStringsInnerLayer[layerID-1] < stringID) {
      tidStringsInnerLayer[layerID-1] = stringID;
    }
  } else {
    if (tidStringsOuterLayer[layerID-1] < stringID) {
      tidStringsOuterLayer[layerID-1] = stringID;
    }
  }

  tibSideIDs.  insert(sideID);
  tibLayerIDs. insert(layerID);
  tibStringIDs.insert(stringID);
  tibModuleIDs.insert(moduleID);
}

//_____________________________________________________________________________
void TrackerAlignmentLevelBuilder
::addTIDDetUnitInfo(const DetId& detId)
{
  auto sideID   = trackerTopology->tidSide(detId);
  auto wheelID  = trackerTopology->tidWheel(detId);
  auto ringID   = trackerTopology->tidRing(detId);
  auto moduleID = trackerTopology->module(detId);

  // tidOrder
  tidSideIDs.  insert(sideID);
  tidWheelIDs. insert(wheelID);
  tidRingIDs.  insert(ringID);
  tidModuleIDs.insert(moduleID);
}

//_____________________________________________________________________________
void TrackerAlignmentLevelBuilder
::addTOBDetUnitInfo(const DetId& detId)
{
  auto layerID  = trackerTopology->tobLayer(detId);
  auto sideID   = trackerTopology->tobSide(detId);
  auto rodID    = trackerTopology->tobRod(detId);
  auto moduleID = trackerTopology->module(detId);

  tobLayerIDs. insert(layerID);
  tobSideIDs.  insert(sideID);
  tobRodIDs.   insert(rodID);
  tobModuleIDs.insert(moduleID);
}

//_____________________________________________________________________________
void TrackerAlignmentLevelBuilder
::addTECDetUnitInfo(const DetId& detId)
{
  auto sideID   = trackerTopology->tecSide(detId);
  auto wheelID  = trackerTopology->tecWheel(detId);
  auto petalID  = trackerTopology->tecPetalNumber(detId);
  auto ringID   = trackerTopology->tecRing(detId);
  auto moduleID = trackerTopology->module(detId);

  tecSideIDs.  insert(sideID);
  tecWheelIDs. insert(wheelID);
  tecPetalIDs. insert(petalID);
  tecRingIDs.  insert(ringID);
  tecModuleIDs.insert(moduleID);
}



//_____________________________________________________________________________
void TrackerAlignmentLevelBuilder
::buildPXBAlignmentLevels()
{
  int maxNumModules = pxbModuleIDs.size();
  int maxNumLadders = pxbLadderIDs.size() / 2; // divide by 2 since we have
                                               // HalfBarrels
  int maxNumLayers  = pxbLayerIDs.size();

  std::ostringstream ss;
  ss << "determined following numbers for "
     << AlignableObjectId::idToString(align::TPBBarrel) << " geometry:" << "\n"
     << "   max. number of modules: " << maxNumModules                  << "\n"
     << "   max. number of ladders: " << maxNumLadders                  << "\n";

  for (size_t layer = 0; layer < pxbLaddersPerLayer.size(); ++layer) {
    // divide by 4, because we need the ladders per quarter cylinder
    align::tpb::lpqc.push_back(pxbLaddersPerLayer[layer] / 4);
    ss << "      ladders in layer-" << layer << ": "
       << pxbLaddersPerLayer[layer] << "\n";
  }

  ss << "   max. number of layers:  " << maxNumLayers;
  edm::LogInfo("AlignableBuildProcess")
     << "@SUB=TrackerAlignmentLevelBuilder::buildPXBAlignmentLevels"
     << ss.str();

  pxb.push_back(new AlignmentLevel(align::TPBModule,     maxNumModules, false));
  pxb.push_back(new AlignmentLevel(align::TPBLadder,     maxNumLadders, true));
  pxb.push_back(new AlignmentLevel(align::TPBLayer,      maxNumLayers,  false));
  pxb.push_back(new AlignmentLevel(align::TPBHalfBarrel, 2,             false));
  pxb.push_back(new AlignmentLevel(align::TPBBarrel,     1,             false));
}

//_____________________________________________________________________________
void TrackerAlignmentLevelBuilder
::buildPXEAlignmentLevels()
{
  int maxNumModules = pxeModuleIDs.size();
  int maxNumPanels  = pxePanelIDs.size();
  int maxNumBlades  = pxeBladeIDs.size() / 2;
  int maxNumDisks   = pxeDiskIDs.size();
  int maxNumSides   = pxeSideIDs.size();

  std::ostringstream ss;
  ss << "determined following numbers for "
     << AlignableObjectId::idToString(align::TPEEndcap) << " geometry:" << "\n"
     << "   max. number of modules: " << maxNumModules                  << "\n"
     << "   max. number of panels:  " << maxNumPanels                   << "\n"
     << "   max. number of blades:  " << maxNumBlades                   << "\n";

  align::tpe::bpqd = maxNumBlades / 2;

  ss << "      blades per quarter disk: " << align::tpe::bpqd << "\n"
     << "   max. number of disks:   " << maxNumDisks                    << "\n"
     << "   max. number of sides:   " << maxNumSides;
  edm::LogInfo("AlignableBuildProcess")
     << "@SUB=TrackerAlignmentLevelBuilder::buildPXEAlignmentLevels"
     << ss.str();

  pxe.push_back(new AlignmentLevel(align::TPEModule,       maxNumModules, false));
  pxe.push_back(new AlignmentLevel(align::TPEPanel,        maxNumPanels,  true));
  pxe.push_back(new AlignmentLevel(align::TPEBlade,        maxNumBlades,  true));
  pxe.push_back(new AlignmentLevel(align::TPEHalfDisk,     maxNumDisks,   false));
  pxe.push_back(new AlignmentLevel(align::TPEHalfCylinder, 2,             false));
  pxe.push_back(new AlignmentLevel(align::TPEEndcap,       maxNumSides,   false));
}

//_____________________________________________________________________________
void TrackerAlignmentLevelBuilder
::buildTIBAlignmentLevels()
{
  int maxNumModules = tibModuleIDs.size();
  int maxNumStrings = tibStringIDs.size();
  int maxNumLayers  = tibLayerIDs.size();
  int maxNumSides   = tibSideIDs.size();

  std::ostringstream ss;
  ss << "determined following numbers for "
     << AlignableObjectId::idToString(align::TIBBarrel) << " geometry:" << "\n"
     << "   max. number of modules: " << maxNumModules                  << "\n"
     << "   max. number of strings: " << maxNumStrings                  << "\n";

  for (size_t layer = 0; layer < tidStringsInnerLayer.size(); ++layer) {
    // divide by 2, because we have HalfShells
    align::tib::sphs.push_back(tidStringsInnerLayer[layer] / 2);
    align::tib::sphs.push_back(tidStringsOuterLayer[layer] / 2);

    ss << "      strings in layer-" << layer << " (inside):  "
       << tidStringsInnerLayer[layer] << "\n"
       << "      strings in layer-" << layer << " (outside): "
       << tidStringsOuterLayer[layer] << "\n";
  }

  ss << "   max. number of layers:  " << maxNumLayers                   << "\n"
     << "   max. number of sides:   " << maxNumSides;
  edm::LogInfo("AlignableBuildProcess")
       << "@SUB=TrackerAlignmentLevelBuilder::buildTIBAlignmentLevels"
       << ss.str();

  tib.push_back(new AlignmentLevel(align::TIBModule,     maxNumModules, false));
  tib.push_back(new AlignmentLevel(align::TIBString,     maxNumStrings, true));
  tib.push_back(new AlignmentLevel(align::TIBSurface,    2, false)); // 2 surfaces per half shell
  tib.push_back(new AlignmentLevel(align::TIBHalfShell,  2, false)); // 2 half shells per layer
  tib.push_back(new AlignmentLevel(align::TIBLayer,      maxNumLayers, false));
  tib.push_back(new AlignmentLevel(align::TIBHalfBarrel, 2, false));
  tib.push_back(new AlignmentLevel(align::TIBBarrel,     1, false));
}

//_____________________________________________________________________________
void TrackerAlignmentLevelBuilder
::buildTIDAlignmentLevels()
{
  int maxNumModules = tidModuleIDs.size();
  int maxNumRings   = tidRingIDs.size();
  // TODO: for PhaseII geometry the method name for tidWheel changes:
  //       -> trackerTopology->tidDisk(detId);
  int maxNumWheels  = tidWheelIDs.size();
  int maxNumSides   = tidSideIDs.size();

  edm::LogInfo("AlignableBuildProcess")
     << "@SUB=TrackerAlignmentLevelBuilder::buildTIDAlignmentLevels"
     << "determined following numbers for "
     << AlignableObjectId::idToString(align::TIDEndcap) << " geometry:" << "\n"
     << "   max. number of modules: " << maxNumModules                  << "\n"
     << "   max. number of rings:   " << maxNumRings                    << "\n"
     << "   max. number of wheels:  " << maxNumWheels                   << "\n"
     << "   max. number of sides:   " << maxNumSides;

  tid.push_back(new AlignmentLevel(align::TIDModule, maxNumModules, false));
  tid.push_back(new AlignmentLevel(align::TIDSide,   2,             false)); // 2 sides per ring
  tid.push_back(new AlignmentLevel(align::TIDRing,   maxNumRings,   false));
  tid.push_back(new AlignmentLevel(align::TIDDisk,   maxNumWheels,  false));
  tid.push_back(new AlignmentLevel(align::TIDEndcap, 2,             false)); // 2 endcaps in TID
}

//_____________________________________________________________________________
void TrackerAlignmentLevelBuilder
::buildTOBAlignmentLevels()
{
  int maxNumModules = tobModuleIDs.size();
  int maxNumRods    = tobRodIDs.size();
  int maxNumSides   = tobSideIDs.size();
  int maxNumLayers  = tobLayerIDs.size();

  edm::LogInfo("AlignableBuildProcess")
     << "@SUB=TrackerAlignmentLevelBuilder::buildTOBAlignmentLevels"
     << "determined following numbers for "
     << AlignableObjectId::idToString(align::TOBBarrel) << " geometry:" << "\n"
     << "   max. number of modules: " << maxNumModules                  << "\n"
     << "   max. number of rods:    " << maxNumRods                     << "\n"
     << "   max. number of sides:   " << maxNumSides                    << "\n"
     << "   max. number of layers:  " << maxNumLayers;

  tob.push_back(new AlignmentLevel(align::TOBModule,     maxNumModules, false));
  tob.push_back(new AlignmentLevel(align::TOBRod,        maxNumRods,    true));
  tob.push_back(new AlignmentLevel(align::TOBLayer,      maxNumLayers,  false));
  tob.push_back(new AlignmentLevel(align::TOBHalfBarrel, maxNumSides,   false));
  tob.push_back(new AlignmentLevel(align::TOBBarrel,     1,             false));
}

//_____________________________________________________________________________
void TrackerAlignmentLevelBuilder
::buildTECAlignmentLevels()
{
  int maxNumModules = tecModuleIDs.size();
  int maxNumRings   = tecRingIDs.size();
  int maxNumPetals  = tecPetalIDs.size();
  int maxNumDisks   = tecWheelIDs.size();
  int maxNumSides   = tecSideIDs.size();

  edm::LogInfo("AlignableBuildProcess")
     << "@SUB=TrackerAlignmentLevelBuilder::buildTECAlignmentLevels"
     << "determined following numbers for "
     << AlignableObjectId::idToString(align::TECEndcap) << " geometry:" << "\n"
     << "   max. number of modules: " << maxNumModules                  << "\n"
     << "   max. number of rings:   " << maxNumRings                    << "\n"
     << "   max. number of petals:  " << maxNumPetals                   << "\n"
     << "   max. number of wheels:  " << maxNumDisks                    << "\n"
     << "   max. number of sides:   " << maxNumSides;

  tec.push_back(new AlignmentLevel(align::TECModule, maxNumModules, false));
  tec.push_back(new AlignmentLevel(align::TECRing,   maxNumRings,   true));
  tec.push_back(new AlignmentLevel(align::TECPetal,  maxNumPetals,  true));
  tec.push_back(new AlignmentLevel(align::TECSide,   2,             false)); // 2 sides per disk
  tec.push_back(new AlignmentLevel(align::TECDisk,   maxNumDisks,   false));
  tec.push_back(new AlignmentLevel(align::TECEndcap, 2,             false));
}
