
#include "Alignment/TrackerAlignment/interface/TrackerAlignmentLevelBuilder.h"

// Original Author:  Max Stark
//         Created:  Wed, 10 Feb 2016 13:48:41 CET

// topology and geometry
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"



//=============================================================================
//===   PUBLIC METHOD IMPLEMENTATION                                        ===
//=============================================================================

//_____________________________________________________________________________
TrackerAlignmentLevelBuilder
::TrackerAlignmentLevelBuilder(const TrackerTopology* trackerTopology,
			       const TrackerGeometry* trackerGeometry) :
  trackerTopology_(trackerTopology),
  alignableObjectId_(trackerGeometry, nullptr, nullptr),
  trackerNameSpace_(trackerTopology)
{
}

//_____________________________________________________________________________
TrackerAlignmentLevelBuilder::
~TrackerAlignmentLevelBuilder()
{
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
std::vector<align::AlignmentLevels> TrackerAlignmentLevelBuilder
::build()
{
  std::vector<align::AlignmentLevels> levels;
  levels.push_back(buildPXBAlignmentLevels());
  levels.push_back(buildPXEAlignmentLevels());
  levels.push_back(buildTIBAlignmentLevels());
  levels.push_back(buildTIDAlignmentLevels());
  levels.push_back(buildTOBAlignmentLevels());
  levels.push_back(buildTECAlignmentLevels());
  levelsBuilt_ = true;
  return levels;
}


//______________________________________________________________________________
const align::TrackerNameSpace& TrackerAlignmentLevelBuilder
::trackerNameSpace() const {
  if (levelsBuilt_) {
    return trackerNameSpace_;
  } else {
    throw cms::Exception("LogicError")
      << "@SUB=TrackerAlignmentLevelBuilder::trackerNameSpace\n"
      << "trying to get the name space before it has been properly initialized;"
      << " please call TrackerAlignmentLevelBuilder::build() first";
  }
}



//=============================================================================
//===   PRIVATE METHOD IMPLEMENTATION                                       ===
//=============================================================================

//_____________________________________________________________________________
void TrackerAlignmentLevelBuilder
::addPXBDetUnitInfo(const DetId& detId)
{
  auto layerID  = trackerTopology_->pxbLayer(detId);
  auto ladderID = trackerTopology_->pxbLadder(detId);
  auto moduleID = trackerTopology_->module(detId);

  if (pxbLaddersPerLayer_[layerID-1] < ladderID) {
    pxbLaddersPerLayer_[layerID-1] = ladderID;
  }

  pxbLayerIDs_. insert(layerID);
  pxbLadderIDs_.insert(ladderID);
  pxbModuleIDs_.insert(moduleID);
}

//_____________________________________________________________________________
void TrackerAlignmentLevelBuilder
::addPXEDetUnitInfo(const DetId& detId)
{
  auto sideID   = trackerTopology_->pxfSide(detId);
  auto diskID   = trackerTopology_->pxfDisk(detId);
  auto bladeID  = trackerTopology_->pxfBlade(detId);
  auto panelID  = trackerTopology_->pxfPanel(detId);
  auto moduleID = trackerTopology_->module(detId);

  pxeSideIDs_.  insert(sideID);
  pxeDiskIDs_.  insert(diskID);
  pxeBladeIDs_. insert(bladeID);
  pxePanelIDs_. insert(panelID);
  pxeModuleIDs_.insert(moduleID);
}

//_____________________________________________________________________________
void TrackerAlignmentLevelBuilder
::addTIBDetUnitInfo(const DetId& detId)
{
  auto sideID    = trackerTopology_->tibSide(detId);
  auto layerID   = trackerTopology_->tibLayer(detId);
  auto layerSide = trackerTopology_->tibOrder(detId);
  auto stringID  = trackerTopology_->tibString(detId);
  auto moduleID  = trackerTopology_->module(detId);

  if (layerSide == 1) {
    if (tidStringsInnerLayer_[layerID-1] < stringID) {
      tidStringsInnerLayer_[layerID-1] = stringID;
    }
  } else {
    if (tidStringsOuterLayer_[layerID-1] < stringID) {
      tidStringsOuterLayer_[layerID-1] = stringID;
    }
  }

  tibSideIDs_.  insert(sideID);
  tibLayerIDs_. insert(layerID);
  tibStringIDs_.insert(stringID);
  tibModuleIDs_.insert(moduleID);
}

//_____________________________________________________________________________
void TrackerAlignmentLevelBuilder
::addTIDDetUnitInfo(const DetId& detId)
{
  auto sideID   = trackerTopology_->tidSide(detId);
  auto wheelID  = trackerTopology_->tidWheel(detId);
  auto ringID   = trackerTopology_->tidRing(detId);
  auto moduleID = trackerTopology_->module(detId);

  // tidOrder
  tidSideIDs_.  insert(sideID);
  tidWheelIDs_. insert(wheelID);
  tidRingIDs_.  insert(ringID);
  tidModuleIDs_.insert(moduleID);
}

//_____________________________________________________________________________
void TrackerAlignmentLevelBuilder
::addTOBDetUnitInfo(const DetId& detId)
{
  auto layerID  = trackerTopology_->tobLayer(detId);
  auto sideID   = trackerTopology_->tobSide(detId);
  auto rodID    = trackerTopology_->tobRod(detId);
  auto moduleID = trackerTopology_->module(detId);

  tobLayerIDs_. insert(layerID);
  tobSideIDs_.  insert(sideID);
  tobRodIDs_.   insert(rodID);
  tobModuleIDs_.insert(moduleID);
}

//_____________________________________________________________________________
void TrackerAlignmentLevelBuilder
::addTECDetUnitInfo(const DetId& detId)
{
  auto sideID   = trackerTopology_->tecSide(detId);
  auto wheelID  = trackerTopology_->tecWheel(detId);
  auto petalID  = trackerTopology_->tecPetalNumber(detId);
  auto ringID   = trackerTopology_->tecRing(detId);
  auto moduleID = trackerTopology_->module(detId);

  tecSideIDs_.  insert(sideID);
  tecWheelIDs_. insert(wheelID);
  tecPetalIDs_. insert(petalID);
  tecRingIDs_.  insert(ringID);
  tecModuleIDs_.insert(moduleID);
}



//_____________________________________________________________________________
align::AlignmentLevels TrackerAlignmentLevelBuilder
::buildPXBAlignmentLevels()
{
  int maxNumModules = pxbModuleIDs_.size();
  int maxNumLadders = pxbLadderIDs_.size() / 2; // divide by 2 since we have
                                               // HalfBarrels
  int maxNumLayers  = pxbLayerIDs_.size();

  std::ostringstream ss;
  ss << "determined following numbers for "
     << alignableObjectId_.idToString(align::TPBBarrel) << " geometry:" << "\n"
     << "   max. number of modules: " << maxNumModules                  << "\n"
     << "   max. number of ladders: " << maxNumLadders                  << "\n";

  for (size_t layer = 0; layer < pxbLaddersPerLayer_.size(); ++layer) {
    // divide by 4, because we need the ladders per quarter cylinder
    trackerNameSpace_.tpb_.lpqc_.push_back(pxbLaddersPerLayer_[layer] / 4);
    ss << "      ladders in layer-" << layer << ": "
       << pxbLaddersPerLayer_[layer] << "\n";
  }

  ss << "   max. number of layers:  " << maxNumLayers;
  edm::LogInfo("AlignableBuildProcess")
     << "@SUB=TrackerAlignmentLevelBuilder::buildPXBAlignmentLevels"
     << ss.str();

  align::AlignmentLevels pxb;
  pxb.push_back(std::make_unique<AlignmentLevel>(align::TPBModule,     maxNumModules, false));
  pxb.push_back(std::make_unique<AlignmentLevel>(align::TPBLadder,     maxNumLadders, true));
  pxb.push_back(std::make_unique<AlignmentLevel>(align::TPBLayer,      maxNumLayers,  false));
  pxb.push_back(std::make_unique<AlignmentLevel>(align::TPBHalfBarrel, 2,             false));
  pxb.push_back(std::make_unique<AlignmentLevel>(align::TPBBarrel,     1,             false));
  return pxb;
}

//_____________________________________________________________________________
align::AlignmentLevels TrackerAlignmentLevelBuilder
::buildPXEAlignmentLevels()
{
  int maxNumModules = pxeModuleIDs_.size();
  int maxNumPanels  = pxePanelIDs_.size();
  int maxNumBlades  = pxeBladeIDs_.size() / 2;
  int maxNumDisks   = pxeDiskIDs_.size();
  int maxNumSides   = pxeSideIDs_.size();

  std::ostringstream ss;
  ss << "determined following numbers for "
     << alignableObjectId_.idToString(align::TPEEndcap) << " geometry:" << "\n"
     << "   max. number of modules: " << maxNumModules                  << "\n"
     << "   max. number of panels:  " << maxNumPanels                   << "\n"
     << "   max. number of blades:  " << maxNumBlades                   << "\n";

  trackerNameSpace_.tpe_.bpqd_ = maxNumBlades / 2;

  ss << "      blades per quarter disk: " << trackerNameSpace_.tpe_.bpqd_ << "\n"
     << "   max. number of disks:   " << maxNumDisks                    << "\n"
     << "   max. number of sides:   " << maxNumSides;
  edm::LogInfo("AlignableBuildProcess")
     << "@SUB=TrackerAlignmentLevelBuilder::buildPXEAlignmentLevels"
     << ss.str();

  align::AlignmentLevels pxe;
  pxe.push_back(std::make_unique<AlignmentLevel>(align::TPEModule,       maxNumModules, false));
  pxe.push_back(std::make_unique<AlignmentLevel>(align::TPEPanel,        maxNumPanels,  true));
  pxe.push_back(std::make_unique<AlignmentLevel>(align::TPEBlade,        maxNumBlades,  true));
  pxe.push_back(std::make_unique<AlignmentLevel>(align::TPEHalfDisk,     maxNumDisks,   false));
  pxe.push_back(std::make_unique<AlignmentLevel>(align::TPEHalfCylinder, 2,             false));
  pxe.push_back(std::make_unique<AlignmentLevel>(align::TPEEndcap,       maxNumSides,   false));
  return pxe;
}

//_____________________________________________________________________________
align::AlignmentLevels TrackerAlignmentLevelBuilder
::buildTIBAlignmentLevels()
{
  int maxNumModules = tibModuleIDs_.size();
  int maxNumStrings = tibStringIDs_.size();
  int maxNumLayers  = tibLayerIDs_.size();
  int maxNumSides   = tibSideIDs_.size();

  std::ostringstream ss;
  ss << "determined following numbers for "
     << alignableObjectId_.idToString(align::TIBBarrel) << " geometry:" << "\n"
     << "   max. number of modules: " << maxNumModules                  << "\n"
     << "   max. number of strings: " << maxNumStrings                  << "\n";

  for (size_t layer = 0; layer < tidStringsInnerLayer_.size(); ++layer) {
    // divide by 2, because we have HalfShells
    trackerNameSpace_.tib_.sphs_.push_back(tidStringsInnerLayer_[layer] / 2);
    trackerNameSpace_.tib_.sphs_.push_back(tidStringsOuterLayer_[layer] / 2);

    ss << "      strings in layer-" << layer << " (inside):  "
       << tidStringsInnerLayer_[layer] << "\n"
       << "      strings in layer-" << layer << " (outside): "
       << tidStringsOuterLayer_[layer] << "\n";
  }

  ss << "   max. number of layers:  " << maxNumLayers                   << "\n"
     << "   max. number of sides:   " << maxNumSides;
  edm::LogInfo("AlignableBuildProcess")
       << "@SUB=TrackerAlignmentLevelBuilder::buildTIBAlignmentLevels"
       << ss.str();

  align::AlignmentLevels tib;
  tib.push_back(std::make_unique<AlignmentLevel>(align::TIBModule,     maxNumModules, false));
  tib.push_back(std::make_unique<AlignmentLevel>(align::TIBString,     maxNumStrings, true));
  tib.push_back(std::make_unique<AlignmentLevel>(align::TIBSurface,    2, false)); // 2 surfaces per half shell
  tib.push_back(std::make_unique<AlignmentLevel>(align::TIBHalfShell,  2, false)); // 2 half shells per layer
  tib.push_back(std::make_unique<AlignmentLevel>(align::TIBLayer,      maxNumLayers, false));
  tib.push_back(std::make_unique<AlignmentLevel>(align::TIBHalfBarrel, 2, false));
  tib.push_back(std::make_unique<AlignmentLevel>(align::TIBBarrel,     1, false));
  return tib;
}

//_____________________________________________________________________________
align::AlignmentLevels TrackerAlignmentLevelBuilder
::buildTIDAlignmentLevels()
{
  int maxNumModules = tidModuleIDs_.size();
  int maxNumRings   = tidRingIDs_.size();
  // TODO: for PhaseII geometry the method name for tidWheel changes:
  //       -> trackerTopology->tidDisk(detId);
  int maxNumWheels  = tidWheelIDs_.size();
  int maxNumSides   = tidSideIDs_.size();

  edm::LogInfo("AlignableBuildProcess")
     << "@SUB=TrackerAlignmentLevelBuilder::buildTIDAlignmentLevels"
     << "determined following numbers for "
     << alignableObjectId_.idToString(align::TIDEndcap) << " geometry:" << "\n"
     << "   max. number of modules: " << maxNumModules                  << "\n"
     << "   max. number of rings:   " << maxNumRings                    << "\n"
     << "   max. number of wheels:  " << maxNumWheels                   << "\n"
     << "   max. number of sides:   " << maxNumSides;

  align::AlignmentLevels tid;
  tid.push_back(std::make_unique<AlignmentLevel>(align::TIDModule, maxNumModules, false));
  tid.push_back(std::make_unique<AlignmentLevel>(align::TIDSide,   2,             false)); // 2 sides per ring
  tid.push_back(std::make_unique<AlignmentLevel>(align::TIDRing,   maxNumRings,   false));
  tid.push_back(std::make_unique<AlignmentLevel>(align::TIDDisk,   maxNumWheels,  false));
  tid.push_back(std::make_unique<AlignmentLevel>(align::TIDEndcap, 2,             false)); // 2 endcaps in TID
  return tid;
}

//_____________________________________________________________________________
align::AlignmentLevels TrackerAlignmentLevelBuilder
::buildTOBAlignmentLevels()
{
  int maxNumModules = tobModuleIDs_.size();
  int maxNumRods    = tobRodIDs_.size();
  int maxNumSides   = tobSideIDs_.size();
  int maxNumLayers  = tobLayerIDs_.size();

  edm::LogInfo("AlignableBuildProcess")
     << "@SUB=TrackerAlignmentLevelBuilder::buildTOBAlignmentLevels"
     << "determined following numbers for "
     << alignableObjectId_.idToString(align::TOBBarrel) << " geometry:" << "\n"
     << "   max. number of modules: " << maxNumModules                  << "\n"
     << "   max. number of rods:    " << maxNumRods                     << "\n"
     << "   max. number of sides:   " << maxNumSides                    << "\n"
     << "   max. number of layers:  " << maxNumLayers;

  align::AlignmentLevels tob;
  tob.push_back(std::make_unique<AlignmentLevel>(align::TOBModule,     maxNumModules, false));
  tob.push_back(std::make_unique<AlignmentLevel>(align::TOBRod,        maxNumRods,    true));
  tob.push_back(std::make_unique<AlignmentLevel>(align::TOBLayer,      maxNumLayers,  false));
  tob.push_back(std::make_unique<AlignmentLevel>(align::TOBHalfBarrel, maxNumSides,   false));
  tob.push_back(std::make_unique<AlignmentLevel>(align::TOBBarrel,     1,             false));
  return tob;
}

//_____________________________________________________________________________
align::AlignmentLevels TrackerAlignmentLevelBuilder
::buildTECAlignmentLevels()
{
  int maxNumModules = tecModuleIDs_.size();
  int maxNumRings   = tecRingIDs_.size();
  int maxNumPetals  = tecPetalIDs_.size();
  int maxNumDisks   = tecWheelIDs_.size();
  int maxNumSides   = tecSideIDs_.size();

  edm::LogInfo("AlignableBuildProcess")
     << "@SUB=TrackerAlignmentLevelBuilder::buildTECAlignmentLevels"
     << "determined following numbers for "
     << alignableObjectId_.idToString(align::TECEndcap) << " geometry:" << "\n"
     << "   max. number of modules: " << maxNumModules                  << "\n"
     << "   max. number of rings:   " << maxNumRings                    << "\n"
     << "   max. number of petals:  " << maxNumPetals                   << "\n"
     << "   max. number of wheels:  " << maxNumDisks                    << "\n"
     << "   max. number of sides:   " << maxNumSides;

  align::AlignmentLevels tec;
  tec.push_back(std::make_unique<AlignmentLevel>(align::TECModule, maxNumModules, false));
  tec.push_back(std::make_unique<AlignmentLevel>(align::TECRing,   maxNumRings,   true));
  tec.push_back(std::make_unique<AlignmentLevel>(align::TECPetal,  maxNumPetals,  true));
  tec.push_back(std::make_unique<AlignmentLevel>(align::TECSide,   2,             false)); // 2 sides per disk
  tec.push_back(std::make_unique<AlignmentLevel>(align::TECDisk,   maxNumDisks,   false));
  tec.push_back(std::make_unique<AlignmentLevel>(align::TECEndcap, 2,             false));
  return tec;
}
