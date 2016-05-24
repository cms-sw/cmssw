#include "FWCore/Utilities/interface/Exception.h"

#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include <algorithm>

using namespace align;

namespace {
  struct entry {
    StructureType type; 
    const char* name;
  };

  entry entries_RunI[] {
    { invalid         , "invalid"},
    { AlignableDetUnit, "DetUnit"},
    { AlignableDet    , "Det"},

    {TPBModule      , "TPBModule"},
    {TPBLadder      , "TPBLadder"},
    {TPBLayer       , "TPBLayer"},
    {TPBHalfBarrel  , "TPBHalfBarrel"},
    {TPBBarrel      , "TPBBarrel"},

    {TPEModule      , "TPEModule"},
    {TPEPanel       , "TPEPanel"},
    {TPEBlade       , "TPEBlade"},
    {TPEHalfDisk    , "TPEHalfDisk"},
    {TPEHalfCylinder, "TPEHalfCylinder"},
    {TPEEndcap      , "TPEEndcap"},

    {TIBModule      , "TIBModule"},
    {TIBString      , "TIBString"},
    {TIBSurface     , "TIBSurface"},
    {TIBHalfShell   , "TIBHalfShell"},
    {TIBLayer       , "TIBLayer"},
    {TIBHalfBarrel  , "TIBHalfBarrel"},
    {TIBBarrel      , "TIBBarrel"},

    {TIDModule      , "TIDModule"},
    {TIDSide        , "TIDSide"},
    {TIDRing        , "TIDRing"},
    {TIDDisk        , "TIDDisk"},
    {TIDEndcap      , "TIDEndcap"},

    {TOBModule      , "TOBModule"},
    {TOBRod         , "TOBRod"},
    {TOBLayer       , "TOBLayer"},
    {TOBHalfBarrel  , "TOBHalfBarrel"},
    {TOBBarrel      , "TOBBarrel"},

    {TECModule      , "TECModule"},
    {TECRing        , "TECRing"},
    {TECPetal       , "TECPetal"},
    {TECSide        , "TECSide"},
    {TECDisk        , "TECDisk"},
    {TECEndcap      , "TECEndcap"},

    {Pixel          , "Pixel"},
    {Strip          , "Strip"},
    {Tracker        , "Tracker"},

    { AlignableDTBarrel    ,  "DTBarrel"},
    { AlignableDTWheel     ,  "DTWheel"},
    { AlignableDTStation   ,  "DTStation"},
    { AlignableDTChamber   ,  "DTChamber"},
    { AlignableDTSuperLayer,  "DTSuperLayer"},
    { AlignableDTLayer     ,  "DTLayer"},
    { AlignableCSCEndcap   ,  "CSCEndcap"},
    { AlignableCSCStation  ,  "CSCStation"},
    { AlignableCSCRing     ,  "CSCRing"},
    { AlignableCSCChamber  ,  "CSCChamber"},
    { AlignableCSCLayer    ,  "CSCLayer"},
    { AlignableMuon        ,  "Muon"},

    { BeamSpot, "BeamSpot"},
    {notfound, 0}
  };

  entry entries_PhaseI[] {
    { invalid         , "invalid"},
    { AlignableDetUnit, "DetUnit"},
    { AlignableDet    , "Det"},

    {TPBModule      , "P1PXBModule"},
    {TPBLadder      , "P1PXBLadder"},
    {TPBLayer       , "P1PXBLayer"},
    {TPBHalfBarrel  , "P1PXBHalfBarrel"},
    {TPBBarrel      , "P1PXBBarrel"},

    {TPEModule      , "P1PXECModule"},
    {TPEPanel       , "P1PXECPanel"},
    {TPEBlade       , "P1PXECBlade"},
    {TPEHalfDisk    , "P1PXECHalfDisk"},
    {TPEHalfCylinder, "P1PXECHalfCylinder"},
    {TPEEndcap      , "P1PXECEndcap"},

    {TIBModule      , "TIBModule"},
    {TIBString      , "TIBString"},
    {TIBSurface     , "TIBSurface"},
    {TIBHalfShell   , "TIBHalfShell"},
    {TIBLayer       , "TIBLayer"},
    {TIBHalfBarrel  , "TIBHalfBarrel"},
    {TIBBarrel      , "TIBBarrel"},

    {TIDModule      , "TIDModule"},
    {TIDSide        , "TIDSide"},
    {TIDRing        , "TIDRing"},
    {TIDDisk        , "TIDDisk"},
    {TIDEndcap      , "TIDEndcap"},

    {TOBModule      , "TOBModule"},
    {TOBRod         , "TOBRod"},
    {TOBLayer       , "TOBLayer"},
    {TOBHalfBarrel  , "TOBHalfBarrel"},
    {TOBBarrel      , "TOBBarrel"},

    {TECModule      , "TECModule"},
    {TECRing        , "TECRing"},
    {TECPetal       , "TECPetal"},
    {TECSide        , "TECSide"},
    {TECDisk        , "TECDisk"},
    {TECEndcap      , "TECEndcap"},

    {Pixel          , "Pixel"},
    {Strip          , "Strip"},
    {Tracker        , "Tracker"},

    { AlignableDTBarrel    ,  "DTBarrel"},
    { AlignableDTWheel     ,  "DTWheel"},
    { AlignableDTStation   ,  "DTStation"},
    { AlignableDTChamber   ,  "DTChamber"},
    { AlignableDTSuperLayer,  "DTSuperLayer"},
    { AlignableDTLayer     ,  "DTLayer"},
    { AlignableCSCEndcap   ,  "CSCEndcap"},
    { AlignableCSCStation  ,  "CSCStation"},
    { AlignableCSCRing     ,  "CSCRing"},
    { AlignableCSCChamber  ,  "CSCChamber"},
    { AlignableCSCLayer    ,  "CSCLayer"},
    { AlignableMuon        ,  "Muon"},

    { BeamSpot, "BeamSpot"},
    {notfound, 0}
  };

  entry entries_PhaseII[] {
    { invalid         , "invalid"},
    { AlignableDetUnit, "DetUnit"},
    { AlignableDet    , "Det"},

    {TPBModule      , "P1PXBModule"},
    {TPBLadder      , "P1PXBLadder"},
    {TPBLayer       , "P1PXBLayer"},
    {TPBHalfBarrel  , "P1PXBHalfBarrel"},
    {TPBBarrel      , "P1PXBBarrel"},

    {TPEModule      , "P2PXECModule"},
    {TPEPanel       , "P2PXECPanel"},
    {TPEBlade       , "P2PXECBlade"},
    {TPEHalfDisk    , "P2PXECHalfDisk"},
    {TPEHalfCylinder, "P2PXECHalfCylinder"},
    {TPEEndcap      , "P2PXECEndcap"},

    // TIB doesn't exit in PhaseII
    {TIBModule      , "TIBModule-INVALID"},
    {TIBString      , "TIBString-INVALID"},
    {TIBSurface     , "TIBSurface-INVALID"},
    {TIBHalfShell   , "TIBHalfShell-INVALID"},
    {TIBLayer       , "TIBLayer-INVALID"},
    {TIBHalfBarrel  , "TIBHalfBarrel-INVALID"},
    {TIBBarrel      , "TIBBarrel-INVALID"},

    {TIDModule      , "P2OTECModule"},
    {TIDSide        , "P2OTECSide"},
    {TIDRing        , "P2OTECRing"},
    {TIDDisk        , "P2OTECDisk"},
    {TIDEndcap      , "P2OTECEndcap"},

    {TOBModule      , "P2OTBModule"},
    {TOBRod         , "P2OTBRod"},
    {TOBLayer       , "P2OTBLayer"},
    {TOBHalfBarrel  , "P2OTBHalfBarrel"},
    {TOBBarrel      , "P2OTBBarrel"},

    // TEC doesn't exit in PhaseII
    {TECModule      , "TECModule-INVALID"},
    {TECRing        , "TECRing-INVALID"},
    {TECPetal       , "TECPetal-INVALID"},
    {TECSide        , "TECSide-INVALID"},
    {TECDisk        , "TECDisk-INVALID"},
    {TECEndcap      , "TECEndcap-INVALID"},

    {Pixel          , "Pixel"},
    {Strip          , "Strip"},
    {Tracker        , "Tracker"},

    { AlignableDTBarrel    ,  "DTBarrel"},
    { AlignableDTWheel     ,  "DTWheel"},
    { AlignableDTStation   ,  "DTStation"},
    { AlignableDTChamber   ,  "DTChamber"},
    { AlignableDTSuperLayer,  "DTSuperLayer"},
    { AlignableDTLayer     ,  "DTLayer"},
    { AlignableCSCEndcap   ,  "CSCEndcap"},
    { AlignableCSCStation  ,  "CSCStation"},
    { AlignableCSCRing     ,  "CSCRing"},
    { AlignableCSCChamber  ,  "CSCChamber"},
    { AlignableCSCLayer    ,  "CSCLayer"},
    { AlignableMuon        ,  "Muon"},

    { BeamSpot, "BeamSpot"},
    {notfound, 0}
  };

  // This pointer points per default to the structure-names of RunI geometry
  // version. If an upgraded geometry is loaded, one can reset the pointer with
  // help of the isPhaseIGeometry() below.
  entry* entries = entries_RunI;



  constexpr bool same(char const *x, char const *y) {
    return !*x && !*y ? true : (*x == *y && same(x+1, y+1));
  }
  
  constexpr char const *objectIdToString(StructureType type,  entry const *entries) {
    return !entries->name ?  0 :
            entries->type == type ? entries->name :
                                    objectIdToString(type, entries+1);
  }

  constexpr enum StructureType stringToObjectId(char const *name,  entry const *entries) {
    return !entries->name             ? invalid :
            same(entries->name, name) ? entries->type :
                                        stringToObjectId(name, entries+1);
  }
}



//_____________________________________________________________________________
void AlignableObjectId
::isRunIGeometry() {
  entries = entries_RunI;
}

//_____________________________________________________________________________
void AlignableObjectId
::isPhaseIGeometry() {
  entries = entries_PhaseI;
}

//_____________________________________________________________________________
void AlignableObjectId
::isPhaseIIGeometry() {
  entries = entries_PhaseII;
}

//_____________________________________________________________________________
StructureType
AlignableObjectId::nameToType( const std::string &name) const
{
  return stringToId(name.c_str());
}

//_____________________________________________________________________________
std::string AlignableObjectId::typeToName( StructureType type ) const
{
  return idToString(type);
}

//_____________________________________________________________________________
const char *AlignableObjectId::idToString(align::StructureType type)
{
  const char *result = objectIdToString(type, entries);

  if (result == 0)
  {
    throw cms::Exception("AlignableObjectIdError")
      << "Unknown alignableObjectId " << type;
  }

  return result;
}

//_____________________________________________________________________________
align::StructureType AlignableObjectId::stringToId(const char *name)
{
  StructureType result = stringToObjectId(name, entries);
  if (result == -1)
  {
    throw cms::Exception("AlignableObjectIdError")
      << "Unknown alignableObjectId " << name;
  }

  return result;
}
