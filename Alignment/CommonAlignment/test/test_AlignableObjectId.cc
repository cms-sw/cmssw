#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include <cassert>
#include <cstring>

int main(int argc, char**argv)
{
  using namespace align;

  AlignableObjectId alignableObjectId{AlignableObjectId::Geometry::General};

  assert(align::invalid == alignableObjectId.stringToId("invalid"));
  assert(align::AlignableDetUnit == alignableObjectId.stringToId("DetUnit"));
  assert(align::AlignableDet == alignableObjectId.stringToId("Det"));

  assert(align::TPBModule == alignableObjectId.stringToId("TPBModule"));
  assert(align::TPBLadder == alignableObjectId.stringToId("TPBLadder"));
  assert(align::TPBLayer == alignableObjectId.stringToId("TPBLayer"));
  assert(align::TPBHalfBarrel == alignableObjectId.stringToId("TPBHalfBarrel"));
  assert(align::TPBBarrel == alignableObjectId.stringToId("TPBBarrel"));

  assert(align::TPEModule == alignableObjectId.stringToId("TPEModule"));
  assert(align::TPEPanel == alignableObjectId.stringToId("TPEPanel"));
  assert(align::TPEBlade == alignableObjectId.stringToId("TPEBlade"));
  assert(align::TPEHalfDisk == alignableObjectId.stringToId("TPEHalfDisk"));
  assert(align::TPEHalfCylinder == alignableObjectId.stringToId("TPEHalfCylinder"));
  assert(align::TPEEndcap == alignableObjectId.stringToId("TPEEndcap"));

  assert(align::TIBModule == alignableObjectId.stringToId("TIBModule"));
  assert(align::TIBString == alignableObjectId.stringToId("TIBString"));
  assert(align::TIBSurface == alignableObjectId.stringToId("TIBSurface"));
  assert(align::TIBHalfShell == alignableObjectId.stringToId("TIBHalfShell"));
  assert(align::TIBLayer == alignableObjectId.stringToId("TIBLayer"));
  assert(align::TIBHalfBarrel == alignableObjectId.stringToId("TIBHalfBarrel"));
  assert(align::TIBBarrel == alignableObjectId.stringToId("TIBBarrel"));

  assert(align::TIDModule == alignableObjectId.stringToId("TIDModule"));
  assert(align::TIDSide == alignableObjectId.stringToId("TIDSide"));
  assert(align::TIDRing == alignableObjectId.stringToId("TIDRing"));
  assert(align::TIDDisk == alignableObjectId.stringToId("TIDDisk"));
  assert(align::TIDEndcap == alignableObjectId.stringToId("TIDEndcap"));

  assert(align::TOBModule == alignableObjectId.stringToId("TOBModule"));
  assert(align::TOBRod == alignableObjectId.stringToId("TOBRod"));
  assert(align::TOBLayer == alignableObjectId.stringToId("TOBLayer"));
  assert(align::TOBHalfBarrel == alignableObjectId.stringToId("TOBHalfBarrel"));
  assert(align::TOBBarrel == alignableObjectId.stringToId("TOBBarrel"));

  assert(align::TECModule == alignableObjectId.stringToId("TECModule"));
  assert(align::TECRing == alignableObjectId.stringToId("TECRing"));
  assert(align::TECPetal == alignableObjectId.stringToId("TECPetal"));
  assert(align::TECSide == alignableObjectId.stringToId("TECSide"));
  assert(align::TECDisk == alignableObjectId.stringToId("TECDisk"));
  assert(align::TECEndcap == alignableObjectId.stringToId("TECEndcap"));

  assert(align::Pixel == alignableObjectId.stringToId("Pixel"));
  assert(align::Strip == alignableObjectId.stringToId("Strip"));
  assert(align::Tracker == alignableObjectId.stringToId("Tracker"));

  assert(!strcmp(alignableObjectId.idToString(align::AlignableDTBarrel), "DTBarrel"));
  assert(!strcmp(alignableObjectId.idToString(align::AlignableDTWheel), "DTWheel"));
  assert(!strcmp(alignableObjectId.idToString(align::AlignableDTStation), "DTStation"));
  assert(!strcmp(alignableObjectId.idToString(align::AlignableDTChamber), "DTChamber"));
  assert(!strcmp(alignableObjectId.idToString(align::AlignableDTSuperLayer), "DTSuperLayer"));
  assert(!strcmp(alignableObjectId.idToString(align::AlignableDTLayer), "DTLayer"));
  assert(!strcmp(alignableObjectId.idToString(align::AlignableCSCEndcap), "CSCEndcap"));
  assert(!strcmp(alignableObjectId.idToString(align::AlignableCSCStation), "CSCStation"));
  assert(!strcmp(alignableObjectId.idToString(align::AlignableCSCRing), "CSCRing"));
  assert(!strcmp(alignableObjectId.idToString(align::AlignableCSCChamber), "CSCChamber"));
  assert(!strcmp(alignableObjectId.idToString(align::AlignableCSCLayer), "CSCLayer"));
  assert(!strcmp(alignableObjectId.idToString(align::AlignableMuon), "Muon"));
  assert(!strcmp(alignableObjectId.idToString(align::BeamSpot), "BeamSpot"));

  assert(!strcmp(alignableObjectId.idToString(align::invalid), "invalid"));
  assert(!strcmp(alignableObjectId.idToString(align::AlignableDetUnit), "DetUnit"));
  assert(!strcmp(alignableObjectId.idToString(align::AlignableDet), "Det"));
  assert(!strcmp(alignableObjectId.idToString(align::TPBModule), "TPBModule"));
  assert(!strcmp(alignableObjectId.idToString(align::TPBLadder), "TPBLadder"));
  assert(!strcmp(alignableObjectId.idToString(align::TPBLayer), "TPBLayer"));
  assert(!strcmp(alignableObjectId.idToString(align::TPBHalfBarrel), "TPBHalfBarrel"));
  assert(!strcmp(alignableObjectId.idToString(align::TPBBarrel), "TPBBarrel"));
  assert(!strcmp(alignableObjectId.idToString(align::TPEModule), "TPEModule"));
  assert(!strcmp(alignableObjectId.idToString(align::TPEPanel), "TPEPanel"));
  assert(!strcmp(alignableObjectId.idToString(align::TPEBlade), "TPEBlade"));
  assert(!strcmp(alignableObjectId.idToString(align::TPEHalfDisk), "TPEHalfDisk"));
  assert(!strcmp(alignableObjectId.idToString(align::TPEHalfCylinder), "TPEHalfCylinder"));
  assert(!strcmp(alignableObjectId.idToString(align::TPEEndcap), "TPEEndcap"));
  assert(!strcmp(alignableObjectId.idToString(align::TIBModule), "TIBModule"));
  assert(!strcmp(alignableObjectId.idToString(align::TIBString), "TIBString"));
  assert(!strcmp(alignableObjectId.idToString(align::TIBSurface), "TIBSurface"));
  assert(!strcmp(alignableObjectId.idToString(align::TIBHalfShell), "TIBHalfShell"));
  assert(!strcmp(alignableObjectId.idToString(align::TIBLayer), "TIBLayer"));
  assert(!strcmp(alignableObjectId.idToString(align::TIBHalfBarrel), "TIBHalfBarrel"));
  assert(!strcmp(alignableObjectId.idToString(align::TIBBarrel), "TIBBarrel"));
  assert(!strcmp(alignableObjectId.idToString(align::TIDModule), "TIDModule"));
  assert(!strcmp(alignableObjectId.idToString(align::TIDSide), "TIDSide"));
  assert(!strcmp(alignableObjectId.idToString(align::TIDRing), "TIDRing"));
  assert(!strcmp(alignableObjectId.idToString(align::TIDDisk), "TIDDisk"));
  assert(!strcmp(alignableObjectId.idToString(align::TIDEndcap), "TIDEndcap"));
  assert(!strcmp(alignableObjectId.idToString(align::TOBModule), "TOBModule"));
  assert(!strcmp(alignableObjectId.idToString(align::TOBRod), "TOBRod"));
  assert(!strcmp(alignableObjectId.idToString(align::TOBLayer), "TOBLayer"));
  assert(!strcmp(alignableObjectId.idToString(align::TOBHalfBarrel), "TOBHalfBarrel"));
  assert(!strcmp(alignableObjectId.idToString(align::TOBBarrel), "TOBBarrel"));
  assert(!strcmp(alignableObjectId.idToString(align::TECModule), "TECModule"));
  assert(!strcmp(alignableObjectId.idToString(align::TECRing), "TECRing"));
  assert(!strcmp(alignableObjectId.idToString(align::TECPetal), "TECPetal"));
  assert(!strcmp(alignableObjectId.idToString(align::TECSide), "TECSide"));
  assert(!strcmp(alignableObjectId.idToString(align::TECDisk), "TECDisk"));
  assert(!strcmp(alignableObjectId.idToString(align::TECEndcap), "TECEndcap"));
  assert(!strcmp(alignableObjectId.idToString(align::Pixel), "Pixel"));
  assert(!strcmp(alignableObjectId.idToString(align::Strip), "Strip"));
  assert(!strcmp(alignableObjectId.idToString(align::Tracker), "Tracker"));
  assert(!strcmp(alignableObjectId.idToString(align::AlignableDTBarrel), "DTBarrel"));
  assert(!strcmp(alignableObjectId.idToString(align::AlignableDTWheel), "DTWheel"));
  assert(!strcmp(alignableObjectId.idToString(align::AlignableDTStation), "DTStation"));
  assert(!strcmp(alignableObjectId.idToString(align::AlignableDTChamber), "DTChamber"));
  assert(!strcmp(alignableObjectId.idToString(align::AlignableDTSuperLayer), "DTSuperLayer"));
  assert(!strcmp(alignableObjectId.idToString(align::AlignableDTLayer), "DTLayer"));
  assert(!strcmp(alignableObjectId.idToString(align::AlignableCSCEndcap), "CSCEndcap"));
  assert(!strcmp(alignableObjectId.idToString(align::AlignableCSCStation), "CSCStation"));
  assert(!strcmp(alignableObjectId.idToString(align::AlignableCSCRing), "CSCRing"));
  assert(!strcmp(alignableObjectId.idToString(align::AlignableCSCChamber), "CSCChamber"));
  assert(!strcmp(alignableObjectId.idToString(align::AlignableCSCLayer), "CSCLayer"));
  assert(!strcmp(alignableObjectId.idToString(align::AlignableMuon), "Muon"));
  assert(!strcmp(alignableObjectId.idToString(align::BeamSpot), "BeamSpot"));
 // assert(notfound== alignableObjectId.stringToId(0));
}
