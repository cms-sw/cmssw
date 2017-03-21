#include "Alignment/TrackerAlignment/interface/TrackerAlignableIndexer.h"


//______________________________________________________________________________
TrackerAlignableIndexer
::TrackerAlignableIndexer(const align::TrackerNameSpace& tns) :
  tns_(tns)
{
  using namespace std::placeholders;
  using namespace align;
  theCounters.clear();

  // Barrel Pixel
  theCounters[TPBModule]     = std::bind(&TrackerNameSpace::TPB::moduleNumber,     &(tns_.tpb()), _1);
  theCounters[TPBLadder]     = std::bind(&TrackerNameSpace::TPB::ladderNumber,     &(tns_.tpb()), _1);
  theCounters[TPBLayer]      = std::bind(&TrackerNameSpace::TPB::layerNumber,      &(tns_.tpb()), _1);
  theCounters[TPBHalfBarrel] = std::bind(&TrackerNameSpace::TPB::halfBarrelNumber, &(tns_.tpb()), _1);
  theCounters[TPBBarrel]     = std::bind(&TrackerNameSpace::TPB::barrelNumber,     &(tns_.tpb()), _1);

  // Forward Pixel
  theCounters[TPEModule]       = std::bind(&TrackerNameSpace::TPE::moduleNumber,       &(tns_.tpe()), _1);
  theCounters[TPEPanel]        = std::bind(&TrackerNameSpace::TPE::panelNumber,        &(tns_.tpe()), _1);
  theCounters[TPEBlade]        = std::bind(&TrackerNameSpace::TPE::bladeNumber,        &(tns_.tpe()), _1);
  theCounters[TPEHalfDisk]     = std::bind(&TrackerNameSpace::TPE::halfDiskNumber,     &(tns_.tpe()), _1);
  theCounters[TPEHalfCylinder] = std::bind(&TrackerNameSpace::TPE::halfCylinderNumber, &(tns_.tpe()), _1);
  theCounters[TPEEndcap]       = std::bind(&TrackerNameSpace::TPE::endcapNumber,       &(tns_.tpe()), _1);

  // Tracker Inner Barrel
  theCounters[TIBModule]     = std::bind(&TrackerNameSpace::TIB::moduleNumber,     &(tns_.tib()), _1);
  theCounters[TIBString]     = std::bind(&TrackerNameSpace::TIB::stringNumber,     &(tns_.tib()), _1);
  theCounters[TIBSurface]    = std::bind(&TrackerNameSpace::TIB::surfaceNumber,    &(tns_.tib()), _1);
  theCounters[TIBHalfShell]  = std::bind(&TrackerNameSpace::TIB::halfShellNumber,  &(tns_.tib()), _1);
  theCounters[TIBLayer]      = std::bind(&TrackerNameSpace::TIB::layerNumber,      &(tns_.tib()), _1);
  theCounters[TIBHalfBarrel] = std::bind(&TrackerNameSpace::TIB::halfBarrelNumber, &(tns_.tib()), _1);
  theCounters[TIBBarrel]     = std::bind(&TrackerNameSpace::TIB::barrelNumber,     &(tns_.tib()), _1);

  // Tracker Inner Disk
  theCounters[TIDModule] = std::bind(&TrackerNameSpace::TID::moduleNumber, &(tns_.tid()), _1);
  theCounters[TIDSide]   = std::bind(&TrackerNameSpace::TID::sideNumber,   &(tns_.tid()), _1);
  theCounters[TIDRing]   = std::bind(&TrackerNameSpace::TID::ringNumber,   &(tns_.tid()), _1);
  theCounters[TIDDisk]   = std::bind(&TrackerNameSpace::TID::diskNumber,   &(tns_.tid()), _1);
  theCounters[TIDEndcap] = std::bind(&TrackerNameSpace::TID::endcapNumber, &(tns_.tid()), _1);

  // Tracker Outer Barrel
  theCounters[TOBModule]     = std::bind(&TrackerNameSpace::TOB::moduleNumber,     &(tns_.tob()), _1);
  theCounters[TOBRod]        = std::bind(&TrackerNameSpace::TOB::rodNumber,        &(tns_.tob()), _1);
  theCounters[TOBLayer]      = std::bind(&TrackerNameSpace::TOB::layerNumber,      &(tns_.tob()), _1);
  theCounters[TOBHalfBarrel] = std::bind(&TrackerNameSpace::TOB::halfBarrelNumber, &(tns_.tob()), _1);
  theCounters[TOBBarrel]     = std::bind(&TrackerNameSpace::TOB::barrelNumber,     &(tns_.tob()), _1);

  // Tracker Endcaps
  theCounters[TECModule] = std::bind(&TrackerNameSpace::TEC::moduleNumber, &(tns_.tec()), _1);
  theCounters[TECRing]   = std::bind(&TrackerNameSpace::TEC::ringNumber,   &(tns_.tec()), _1);
  theCounters[TECPetal]  = std::bind(&TrackerNameSpace::TEC::petalNumber,  &(tns_.tec()), _1);
  theCounters[TECSide]   = std::bind(&TrackerNameSpace::TEC::sideNumber,   &(tns_.tec()), _1);
  theCounters[TECDisk]   = std::bind(&TrackerNameSpace::TEC::diskNumber,   &(tns_.tec()), _1);
  theCounters[TECEndcap] = std::bind(&TrackerNameSpace::TEC::endcapNumber, &(tns_.tec()), _1);
}
