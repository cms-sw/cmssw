#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/CommonAlignment/src/AlignableObjectId.cc"
#include <iostream>
#include <cassert>
#include <cstring>

int main(int argc, char**argv)
{
  assert(invalid ==AlignableObjectId::stringToId("invalid"));
  assert(AlignableDetUnit ==AlignableObjectId::stringToId("DetUnit"));
  assert(AlignableDet ==AlignableObjectId::stringToId("Det"));

  assert(TPBModule == AlignableObjectId::stringToId("TPBModule"));
  assert(TPBLadder == AlignableObjectId::stringToId("TPBLadder"));
  assert(TPBLayer == AlignableObjectId::stringToId("TPBLayer"));
  assert(TPBHalfBarrel == AlignableObjectId::stringToId("TPBHalfBarrel"));
  assert(TPBBarrel == AlignableObjectId::stringToId("TPBBarrel"));

  assert(TPEModule == AlignableObjectId::stringToId("TPEModule"));
  assert(TPEPanel == AlignableObjectId::stringToId("TPEPanel"));
  assert(TPEBlade == AlignableObjectId::stringToId("TPEBlade"));
  assert(TPEHalfDisk == AlignableObjectId::stringToId("TPEHalfDisk"));
  assert(TPEHalfCylinder == AlignableObjectId::stringToId("TPEHalfCylinder"));
  assert(TPEEndcap == AlignableObjectId::stringToId("TPEEndcap"));

  assert(TIBModule == AlignableObjectId::stringToId("TIBModule"));
  assert(TIBString == AlignableObjectId::stringToId("TIBString"));
  assert(TIBSurface == AlignableObjectId::stringToId("TIBSurface"));
  assert(TIBHalfShell == AlignableObjectId::stringToId("TIBHalfShell"));
  assert(TIBLayer == AlignableObjectId::stringToId("TIBLayer"));
  assert(TIBHalfBarrel == AlignableObjectId::stringToId("TIBHalfBarrel"));
  assert(TIBBarrel == AlignableObjectId::stringToId("TIBBarrel"));

  assert(TIDModule == AlignableObjectId::stringToId("TIDModule"));
  assert(TIDSide == AlignableObjectId::stringToId("TIDSide"));
  assert(TIDRing == AlignableObjectId::stringToId("TIDRing"));
  assert(TIDDisk == AlignableObjectId::stringToId("TIDDisk"));
  assert(TIDEndcap == AlignableObjectId::stringToId("TIDEndcap"));

  assert(TOBModule == AlignableObjectId::stringToId("TOBModule"));
  assert(TOBRod == AlignableObjectId::stringToId("TOBRod"));
  assert(TOBLayer == AlignableObjectId::stringToId("TOBLayer"));
  assert(TOBHalfBarrel == AlignableObjectId::stringToId("TOBHalfBarrel"));
  assert(TOBBarrel == AlignableObjectId::stringToId("TOBBarrel"));

  assert(TECModule == AlignableObjectId::stringToId("TECModule"));
  assert(TECRing == AlignableObjectId::stringToId("TECRing"));
  assert(TECPetal == AlignableObjectId::stringToId("TECPetal"));
  assert(TECSide == AlignableObjectId::stringToId("TECSide"));
  assert(TECDisk == AlignableObjectId::stringToId("TECDisk"));
  assert(TECEndcap == AlignableObjectId::stringToId("TECEndcap"));

  assert(Pixel == AlignableObjectId::stringToId("Pixel"));
  assert(Strip == AlignableObjectId::stringToId("Strip"));
  assert(Tracker == AlignableObjectId::stringToId("Tracker"));

  assert(!strcmp(AlignableObjectId::idToString(AlignableDTBarrel), "DTBarrel"));
  assert(!strcmp(AlignableObjectId::idToString(AlignableDTWheel), "DTWheel"));
  assert(!strcmp(AlignableObjectId::idToString(AlignableDTStation), "DTStation"));
  assert(!strcmp(AlignableObjectId::idToString(AlignableDTChamber), "DTChamber"));
  assert(!strcmp(AlignableObjectId::idToString(AlignableDTSuperLayer), "DTSuperLayer"));
  assert(!strcmp(AlignableObjectId::idToString(AlignableDTLayer), "DTLayer"));
  assert(!strcmp(AlignableObjectId::idToString(AlignableCSCEndcap), "CSCEndcap"));
  assert(!strcmp(AlignableObjectId::idToString(AlignableCSCStation), "CSCStation"));
  assert(!strcmp(AlignableObjectId::idToString(AlignableCSCRing), "CSCRing"));
  assert(!strcmp(AlignableObjectId::idToString(AlignableCSCChamber), "CSCChamber"));
  assert(!strcmp(AlignableObjectId::idToString(AlignableCSCLayer), "CSCLayer"));    
  assert(!strcmp(AlignableObjectId::idToString(AlignableMuon), "Muon"));
  assert(!strcmp(AlignableObjectId::idToString(BeamSpot), "BeamSpot"));

  assert(!strcmp(AlignableObjectId::idToString(invalid), "invalid"));
  assert(!strcmp(AlignableObjectId::idToString(AlignableDetUnit), "DetUnit"));
  assert(!strcmp(AlignableObjectId::idToString(AlignableDet), "Det"));
  assert(!strcmp(AlignableObjectId::idToString(TPBModule), "TPBModule"));
  assert(!strcmp(AlignableObjectId::idToString(TPBLadder), "TPBLadder"));
  assert(!strcmp(AlignableObjectId::idToString(TPBLayer), "TPBLayer"));
  assert(!strcmp(AlignableObjectId::idToString(TPBHalfBarrel), "TPBHalfBarrel"));
  assert(!strcmp(AlignableObjectId::idToString(TPBBarrel), "TPBBarrel"));
  assert(!strcmp(AlignableObjectId::idToString(TPEModule), "TPEModule"));
  assert(!strcmp(AlignableObjectId::idToString(TPEPanel), "TPEPanel"));
  assert(!strcmp(AlignableObjectId::idToString(TPEBlade), "TPEBlade"));
  assert(!strcmp(AlignableObjectId::idToString(TPEHalfDisk), "TPEHalfDisk"));
  assert(!strcmp(AlignableObjectId::idToString(TPEHalfCylinder), "TPEHalfCylinder"));
  assert(!strcmp(AlignableObjectId::idToString(TPEEndcap), "TPEEndcap"));
  assert(!strcmp(AlignableObjectId::idToString(TIBModule), "TIBModule"));
  assert(!strcmp(AlignableObjectId::idToString(TIBString), "TIBString"));
  assert(!strcmp(AlignableObjectId::idToString(TIBSurface), "TIBSurface"));
  assert(!strcmp(AlignableObjectId::idToString(TIBHalfShell), "TIBHalfShell"));
  assert(!strcmp(AlignableObjectId::idToString(TIBLayer), "TIBLayer"));
  assert(!strcmp(AlignableObjectId::idToString(TIBHalfBarrel), "TIBHalfBarrel"));
  assert(!strcmp(AlignableObjectId::idToString(TIBBarrel), "TIBBarrel"));
  assert(!strcmp(AlignableObjectId::idToString(TIDModule), "TIDModule"));
  assert(!strcmp(AlignableObjectId::idToString(TIDSide), "TIDSide"));
  assert(!strcmp(AlignableObjectId::idToString(TIDRing), "TIDRing"));
  assert(!strcmp(AlignableObjectId::idToString(TIDDisk), "TIDDisk"));
  assert(!strcmp(AlignableObjectId::idToString(TIDEndcap), "TIDEndcap"));
  assert(!strcmp(AlignableObjectId::idToString(TOBModule), "TOBModule"));
  assert(!strcmp(AlignableObjectId::idToString(TOBRod), "TOBRod"));
  assert(!strcmp(AlignableObjectId::idToString(TOBLayer), "TOBLayer"));
  assert(!strcmp(AlignableObjectId::idToString(TOBHalfBarrel), "TOBHalfBarrel"));
  assert(!strcmp(AlignableObjectId::idToString(TOBBarrel), "TOBBarrel"));
  assert(!strcmp(AlignableObjectId::idToString(TECModule), "TECModule"));
  assert(!strcmp(AlignableObjectId::idToString(TECRing), "TECRing"));
  assert(!strcmp(AlignableObjectId::idToString(TECPetal), "TECPetal"));
  assert(!strcmp(AlignableObjectId::idToString(TECSide), "TECSide"));
  assert(!strcmp(AlignableObjectId::idToString(TECDisk), "TECDisk"));
  assert(!strcmp(AlignableObjectId::idToString(TECEndcap), "TECEndcap"));
  assert(!strcmp(AlignableObjectId::idToString(Pixel), "Pixel"));
  assert(!strcmp(AlignableObjectId::idToString(Strip), "Strip"));
  assert(!strcmp(AlignableObjectId::idToString(Tracker), "Tracker"));
  assert(!strcmp(AlignableObjectId::idToString(AlignableDTBarrel), "DTBarrel"));
  assert(!strcmp(AlignableObjectId::idToString(AlignableDTWheel), "DTWheel"));
  assert(!strcmp(AlignableObjectId::idToString(AlignableDTStation), "DTStation"));
  assert(!strcmp(AlignableObjectId::idToString(AlignableDTChamber), "DTChamber"));
  assert(!strcmp(AlignableObjectId::idToString(AlignableDTSuperLayer), "DTSuperLayer"));
  assert(!strcmp(AlignableObjectId::idToString(AlignableDTLayer), "DTLayer"));
  assert(!strcmp(AlignableObjectId::idToString(AlignableCSCEndcap), "CSCEndcap"));
  assert(!strcmp(AlignableObjectId::idToString(AlignableCSCStation), "CSCStation"));
  assert(!strcmp(AlignableObjectId::idToString(AlignableCSCRing), "CSCRing"));
  assert(!strcmp(AlignableObjectId::idToString(AlignableCSCChamber), "CSCChamber"));
  assert(!strcmp(AlignableObjectId::idToString(AlignableCSCLayer), "CSCLayer"));    
  assert(!strcmp(AlignableObjectId::idToString(AlignableMuon), "Muon"));
  assert(!strcmp(AlignableObjectId::idToString(BeamSpot), "BeamSpot"));
 // assert(notfound== AlignableObjectId::stringToId(0));
}
