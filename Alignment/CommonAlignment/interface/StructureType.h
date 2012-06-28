#ifndef Alignment_CommonAlignment_StructureType_h
#define Alignment_CommonAlignment_StructureType_h

/** \enum StructureType
 *
 *  Enumerate the types of structure an alignable can be.
 *
 *  Basically list the levels in the detector's hierarchy.
 *
 *  $Date: 2010/09/10 10:28:38 $
 *  $Revision: 1.5 $
 *  \author Chung Khim Lae
 */

namespace align
{
  enum StructureType 
  { 
    notfound = -1,
    invalid = 0,
    AlignableDetUnit,
    AlignableDet,

    // Barrel Pixel
    TPBModule,
    TPBLadder,
    TPBLayer, // = 5
    TPBHalfBarrel,
    TPBBarrel,

    // Forward Pixel
    TPEModule,
    TPEPanel,
    TPEBlade, // = 10
    TPEHalfDisk,
    TPEHalfCylinder,
    TPEEndcap,

    // Tracker Inner Barrel
    TIBModule,
    TIBString, // = 15
    TIBSurface,
    TIBHalfShell,
    TIBLayer,
    TIBHalfBarrel,
    TIBBarrel, // = 20

    // Tracker Inner Disks
    TIDModule,
    TIDSide,
    TIDRing,
    TIDDisk,
    TIDEndcap, // = 25

    // Tracker Outer Barrel
    TOBModule,
    TOBRod,
    TOBLayer,
    TOBHalfBarrel,
    TOBBarrel, // = 30

    // Tracker Endcaps
    TECModule,
    TECRing,
    TECPetal,
    TECSide,
    TECDisk, // = 35
    TECEndcap,

    Pixel,
    Strip,
    Tracker, // = 39

    // Muon Detector, not touching these now
    AlignableDTBarrel = 100,
    AlignableDTWheel,
    AlignableDTStation,
    AlignableDTChamber,
    AlignableDTSuperLayer,
    AlignableDTLayer, // = 105
    AlignableCSCEndcap,
    AlignableCSCStation,
    AlignableCSCRing,
    AlignableCSCChamber,
    AlignableCSCLayer, // = 110
    AlignableMuon,

    Detector, // = 112 (what for?)

    Extras = 1000,
    BeamSpot
  };
}

#endif
