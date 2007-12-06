#include "Alignment/TrackerAlignment/interface/TECNameSpace.h"
#include "Alignment/TrackerAlignment/interface/TIBNameSpace.h"
#include "Alignment/TrackerAlignment/interface/TIDNameSpace.h"
#include "Alignment/TrackerAlignment/interface/TOBNameSpace.h"
#include "Alignment/TrackerAlignment/interface/TPBNameSpace.h"
#include "Alignment/TrackerAlignment/interface/TPENameSpace.h"

#include "Alignment/TrackerAlignment/interface/TrackerCounters.h"

using namespace align;

//__________________________________________________________________________________________________
TrackerCounters::TrackerCounters()
{
  theCounters.clear();

  // Barrel Pixel
  theCounters[TPBModule]       = tpb::      moduleNumber;
  theCounters[TPBLadder]       = tpb::      ladderNumber;
  theCounters[TPBLayer]        = tpb::       layerNumber;
  theCounters[TPBHalfBarrel]   = tpb::  halfBarrelNumber;

  // Forward Pixel
  theCounters[TPEModule]       = tpe::      moduleNumber;
  theCounters[TPEPanel]        = tpe::       panelNumber;
  theCounters[TPEBlade]        = tpe::       bladeNumber;
  theCounters[TPEHalfDisk]     = tpe::    halfDiskNumber;
  theCounters[TPEHalfCylinder] = tpe::halfCylinderNumber;
  theCounters[TPEEndcap]       = tpe::      endcapNumber;

  // Tracker Inner Barrel
  theCounters[TIBModule]       = tib::      moduleNumber;
  theCounters[TIBString]       = tib::      stringNumber;
  theCounters[TIBSurface]      = tib::     surfaceNumber;
  theCounters[TIBHalfShell]    = tib::   halfShellNumber;
  theCounters[TIBLayer]        = tib::       layerNumber;
  theCounters[TIBHalfBarrel]   = tib::  halfBarrelNumber;

  // Tracker Inner Barrel
  theCounters[TIDModule]       = tid::      moduleNumber;
  theCounters[TIDSide]         = tid::        sideNumber;
  theCounters[TIDRing]         = tid::        ringNumber;
  theCounters[TIDDisk]         = tid::        diskNumber;
  theCounters[TIDEndcap]       = tid::      endcapNumber;

  // Tracker Outer Barrel
  theCounters[TOBModule]       = tob::      moduleNumber;
  theCounters[TOBRod]          = tob::         rodNumber;
  theCounters[TOBLayer]        = tob::       layerNumber;
  theCounters[TOBHalfBarrel]   = tob::  halfBarrelNumber;

  // Tracker Endcaps
  theCounters[TECModule]       = tec::      moduleNumber;
  theCounters[TECRing]         = tec::        ringNumber;
  theCounters[TECPetal]        = tec::       petalNumber;
  theCounters[TECSide]         = tec::        sideNumber;
  theCounters[TECDisk]         = tec::        diskNumber;
  theCounters[TECEndcap]       = tec::      endcapNumber;
}

