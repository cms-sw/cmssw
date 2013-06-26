#include "FWCore/Utilities/interface/Exception.h"

#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include <algorithm>

using namespace align;

namespace {
  struct entry {
    StructureType type; 
    const char* name;
  };

  constexpr entry entries[]{
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

//__________________________________________________________________________________________________
StructureType
AlignableObjectId::nameToType( const std::string &name) const
{
  return stringToId(name.c_str());
}


//__________________________________________________________________________________________________
std::string AlignableObjectId::typeToName( StructureType type ) const
{
  return idToString(type);
}

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
