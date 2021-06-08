#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"

#include <algorithm>

#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

struct AlignableObjectId::entry {
  align::StructureType type;
  const char *name;
};

namespace {

  constexpr AlignableObjectId::entry entries_RunI[]{{align::invalid, "invalid"},
                                                    {align::AlignableDetUnit, "DetUnit"},
                                                    {align::AlignableDet, "Det"},

                                                    {align::TPBModule, "TPBModule"},
                                                    {align::TPBLadder, "TPBLadder"},
                                                    {align::TPBLayer, "TPBLayer"},
                                                    {align::TPBHalfBarrel, "TPBHalfBarrel"},
                                                    {align::TPBBarrel, "TPBBarrel"},

                                                    {align::TPEModule, "TPEModule"},
                                                    {align::TPEPanel, "TPEPanel"},
                                                    {align::TPEBlade, "TPEBlade"},
                                                    {align::TPEHalfDisk, "TPEHalfDisk"},
                                                    {align::TPEHalfCylinder, "TPEHalfCylinder"},
                                                    {align::TPEEndcap, "TPEEndcap"},

                                                    {align::TIBModule, "TIBModule"},
                                                    {align::TIBString, "TIBString"},
                                                    {align::TIBSurface, "TIBSurface"},
                                                    {align::TIBHalfShell, "TIBHalfShell"},
                                                    {align::TIBLayer, "TIBLayer"},
                                                    {align::TIBHalfBarrel, "TIBHalfBarrel"},
                                                    {align::TIBBarrel, "TIBBarrel"},

                                                    {align::TIDModule, "TIDModule"},
                                                    {align::TIDSide, "TIDSide"},
                                                    {align::TIDRing, "TIDRing"},
                                                    {align::TIDDisk, "TIDDisk"},
                                                    {align::TIDEndcap, "TIDEndcap"},

                                                    {align::TOBModule, "TOBModule"},
                                                    {align::TOBRod, "TOBRod"},
                                                    {align::TOBLayer, "TOBLayer"},
                                                    {align::TOBHalfBarrel, "TOBHalfBarrel"},
                                                    {align::TOBBarrel, "TOBBarrel"},

                                                    {align::TECModule, "TECModule"},
                                                    {align::TECRing, "TECRing"},
                                                    {align::TECPetal, "TECPetal"},
                                                    {align::TECSide, "TECSide"},
                                                    {align::TECDisk, "TECDisk"},
                                                    {align::TECEndcap, "TECEndcap"},

                                                    {align::Pixel, "Pixel"},
                                                    {align::Strip, "Strip"},
                                                    {align::Tracker, "Tracker"},

                                                    {align::AlignableDTBarrel, "DTBarrel"},
                                                    {align::AlignableDTWheel, "DTWheel"},
                                                    {align::AlignableDTStation, "DTStation"},
                                                    {align::AlignableDTChamber, "DTChamber"},
                                                    {align::AlignableDTSuperLayer, "DTSuperLayer"},
                                                    {align::AlignableDTLayer, "DTLayer"},
                                                    {align::AlignableCSCEndcap, "CSCEndcap"},
                                                    {align::AlignableCSCStation, "CSCStation"},
                                                    {align::AlignableCSCRing, "CSCRing"},
                                                    {align::AlignableCSCChamber, "CSCChamber"},
                                                    {align::AlignableCSCLayer, "CSCLayer"},
                                                    {align::AlignableMuon, "Muon"},

                                                    {align::BeamSpot, "BeamSpot"},
                                                    {align::notfound, nullptr}};

  constexpr AlignableObjectId::entry entries_PhaseI[]{{align::invalid, "invalid"},
                                                      {align::AlignableDetUnit, "DetUnit"},
                                                      {align::AlignableDet, "Det"},

                                                      {align::TPBModule, "P1PXBModule"},
                                                      {align::TPBLadder, "P1PXBLadder"},
                                                      {align::TPBLayer, "P1PXBLayer"},
                                                      {align::TPBHalfBarrel, "P1PXBHalfBarrel"},
                                                      {align::TPBBarrel, "P1PXBBarrel"},

                                                      {align::TPEModule, "P1PXECModule"},
                                                      {align::TPEPanel, "P1PXECPanel"},
                                                      {align::TPEBlade, "P1PXECBlade"},
                                                      {align::TPEHalfDisk, "P1PXECHalfDisk"},
                                                      {align::TPEHalfCylinder, "P1PXECHalfCylinder"},
                                                      {align::TPEEndcap, "P1PXECEndcap"},

                                                      {align::TIBModule, "TIBModule"},
                                                      {align::TIBString, "TIBString"},
                                                      {align::TIBSurface, "TIBSurface"},
                                                      {align::TIBHalfShell, "TIBHalfShell"},
                                                      {align::TIBLayer, "TIBLayer"},
                                                      {align::TIBHalfBarrel, "TIBHalfBarrel"},
                                                      {align::TIBBarrel, "TIBBarrel"},

                                                      {align::TIDModule, "TIDModule"},
                                                      {align::TIDSide, "TIDSide"},
                                                      {align::TIDRing, "TIDRing"},
                                                      {align::TIDDisk, "TIDDisk"},
                                                      {align::TIDEndcap, "TIDEndcap"},

                                                      {align::TOBModule, "TOBModule"},
                                                      {align::TOBRod, "TOBRod"},
                                                      {align::TOBLayer, "TOBLayer"},
                                                      {align::TOBHalfBarrel, "TOBHalfBarrel"},
                                                      {align::TOBBarrel, "TOBBarrel"},

                                                      {align::TECModule, "TECModule"},
                                                      {align::TECRing, "TECRing"},
                                                      {align::TECPetal, "TECPetal"},
                                                      {align::TECSide, "TECSide"},
                                                      {align::TECDisk, "TECDisk"},
                                                      {align::TECEndcap, "TECEndcap"},

                                                      {align::Pixel, "Pixel"},
                                                      {align::Strip, "Strip"},
                                                      {align::Tracker, "Tracker"},

                                                      {align::AlignableDTBarrel, "DTBarrel"},
                                                      {align::AlignableDTWheel, "DTWheel"},
                                                      {align::AlignableDTStation, "DTStation"},
                                                      {align::AlignableDTChamber, "DTChamber"},
                                                      {align::AlignableDTSuperLayer, "DTSuperLayer"},
                                                      {align::AlignableDTLayer, "DTLayer"},
                                                      {align::AlignableCSCEndcap, "CSCEndcap"},
                                                      {align::AlignableCSCStation, "CSCStation"},
                                                      {align::AlignableCSCRing, "CSCRing"},
                                                      {align::AlignableCSCChamber, "CSCChamber"},
                                                      {align::AlignableCSCLayer, "CSCLayer"},
                                                      {align::AlignableGEMEndcap, "GEMEndcap"},
                                                      {align::AlignableGEMStation, "GEMStation"},
                                                      {align::AlignableGEMRing, "GEMRing"},
                                                      {align::AlignableGEMSuperChamber, "GEMSuperChamber"},
                                                      {align::AlignableGEMChamber, "GEMChamber"},
                                                      {align::AlignableGEMEtaPartition, "GEMEtaPartition"},
                                                      {align::AlignableMuon, "Muon"},

                                                      {align::BeamSpot, "BeamSpot"},
                                                      {align::notfound, nullptr}};

  constexpr AlignableObjectId::entry entries_PhaseII[]{{align::invalid, "invalid"},
                                                       {align::AlignableDetUnit, "DetUnit"},
                                                       {align::AlignableDet, "Det"},

                                                       {align::TPBModule, "P2PXBModule"},
                                                       {align::TPBLadder, "P2PXBLadder"},
                                                       {align::TPBLayer, "P2PXBLayer"},
                                                       {align::TPBHalfBarrel, "P2PXBHalfBarrel"},
                                                       {align::TPBBarrel, "P2PXBBarrel"},

                                                       {align::TPEModule, "P2PXECModule"},
                                                       {align::TPEPanel, "P2PXECPanel"},
                                                       {align::TPEBlade, "P2PXECBlade"},
                                                       {align::TPEHalfDisk, "P2PXECHalfDisk"},
                                                       {align::TPEHalfCylinder, "P2PXECHalfCylinder"},
                                                       {align::TPEEndcap, "P2PXECEndcap"},

                                                       // TIB doesn't exit in PhaseII
                                                       {align::TIBModule, "TIBModule-INVALID"},
                                                       {align::TIBString, "TIBString-INVALID"},
                                                       {align::TIBSurface, "TIBSurface-INVALID"},
                                                       {align::TIBHalfShell, "TIBHalfShell-INVALID"},
                                                       {align::TIBLayer, "TIBLayer-INVALID"},
                                                       {align::TIBHalfBarrel, "TIBHalfBarrel-INVALID"},
                                                       {align::TIBBarrel, "TIBBarrel-INVALID"},

                                                       {align::TIDModule, "P2OTECModule"},
                                                       {align::TIDSide, "P2OTECSide"},
                                                       {align::TIDRing, "P2OTECRing"},
                                                       {align::TIDDisk, "P2OTECDisk"},
                                                       {align::TIDEndcap, "P2OTECEndcap"},

                                                       {align::TOBModule, "P2OTBModule"},
                                                       {align::TOBRod, "P2OTBRod"},
                                                       {align::TOBLayer, "P2OTBLayer"},
                                                       {align::TOBHalfBarrel, "P2OTBHalfBarrel"},
                                                       {align::TOBBarrel, "P2OTBBarrel"},

                                                       // TEC doesn't exit in PhaseII
                                                       {align::TECModule, "TECModule-INVALID"},
                                                       {align::TECRing, "TECRing-INVALID"},
                                                       {align::TECPetal, "TECPetal-INVALID"},
                                                       {align::TECSide, "TECSide-INVALID"},
                                                       {align::TECDisk, "TECDisk-INVALID"},
                                                       {align::TECEndcap, "TECEndcap-INVALID"},

                                                       {align::Pixel, "Pixel"},
                                                       {align::Strip, "Strip"},
                                                       {align::Tracker, "Tracker"},

                                                       {align::AlignableDTBarrel, "DTBarrel"},
                                                       {align::AlignableDTWheel, "DTWheel"},
                                                       {align::AlignableDTStation, "DTStation"},
                                                       {align::AlignableDTChamber, "DTChamber"},
                                                       {align::AlignableDTSuperLayer, "DTSuperLayer"},
                                                       {align::AlignableDTLayer, "DTLayer"},
                                                       {align::AlignableCSCEndcap, "CSCEndcap"},
                                                       {align::AlignableCSCStation, "CSCStation"},
                                                       {align::AlignableCSCRing, "CSCRing"},
                                                       {align::AlignableCSCChamber, "CSCChamber"},
                                                       {align::AlignableCSCLayer, "CSCLayer"},
                                                       {align::AlignableGEMEndcap, "GEMEndcap"},
                                                       {align::AlignableGEMStation, "GEMStation"},
                                                       {align::AlignableGEMRing, "GEMRing"},
                                                       {align::AlignableGEMSuperChamber, "GEMSuperChamber"},
                                                       {align::AlignableGEMChamber, "GEMChamber"},
                                                       {align::AlignableGEMEtaPartition, "GEMEtaPartition"},
                                                       {align::AlignableMuon, "Muon"},

                                                       {align::BeamSpot, "BeamSpot"},
                                                       {align::notfound, nullptr}};

  constexpr bool same(char const *x, char const *y) { return !*x && !*y ? true : (*x == *y && same(x + 1, y + 1)); }

  constexpr char const *objectIdToString(align::StructureType type, AlignableObjectId::entry const *entries) {
    return !entries->name ? nullptr : entries->type == type ? entries->name : objectIdToString(type, entries + 1);
  }

  constexpr enum align::StructureType stringToObjectId(char const *name, AlignableObjectId::entry const *entries) {
    return !entries->name              ? align::invalid
           : same(entries->name, name) ? entries->type
                                       : stringToObjectId(name, entries + 1);
  }
}  // namespace

//_____________________________________________________________________________
AlignableObjectId ::AlignableObjectId(AlignableObjectId::Geometry geometry) : geometry_(geometry) {
  switch (geometry) {
    case AlignableObjectId::Geometry::RunI:
      entries_ = entries_RunI;
      break;
    case AlignableObjectId::Geometry::PhaseI:
      entries_ = entries_PhaseI;
      break;
    case AlignableObjectId::Geometry::PhaseII:
      entries_ = entries_PhaseII;
      break;
    case AlignableObjectId::Geometry::General:
      entries_ = entries_RunI;
      break;
    case AlignableObjectId::Geometry::Unspecified:
      entries_ = nullptr;
      break;
  }
  if (!entries_) {
    throw cms::Exception("LogicError") << "@SUB=AlignableObjectId::ctor\n"
                                       << "trying to create AlignableObjectId with unspecified geometry";
  }
}

//_____________________________________________________________________________
AlignableObjectId ::AlignableObjectId(const TrackerGeometry *tracker,
                                      const DTGeometry *muonDt,
                                      const CSCGeometry *muonCsc,
                                      const GEMGeometry *muonGem)
    : AlignableObjectId(commonGeometry(trackerGeometry(tracker), muonGeometry(muonDt, muonCsc, muonGem))) {}

//_____________________________________________________________________________
align::StructureType AlignableObjectId::nameToType(const std::string &name) const { return stringToId(name.c_str()); }

//_____________________________________________________________________________
std::string AlignableObjectId::typeToName(align::StructureType type) const { return idToString(type); }

//_____________________________________________________________________________
const char *AlignableObjectId::idToString(align::StructureType type) const {
  const char *result = objectIdToString(type, entries_);

  if (result == nullptr) {
    throw cms::Exception("AlignableObjectIdError") << "Unknown alignableObjectId " << type;
  }

  return result;
}

//_____________________________________________________________________________
align::StructureType AlignableObjectId::stringToId(const char *name) const {
  auto result = stringToObjectId(name, entries_);
  if (result == -1) {
    throw cms::Exception("AlignableObjectIdError") << "Unknown alignableObjectId " << name;
  }

  return result;
}

//______________________________________________________________________________
AlignableObjectId::Geometry AlignableObjectId ::trackerGeometry(const TrackerGeometry *geometry) {
  if (!geometry)
    return Geometry::General;

  if (geometry->isThere(GeomDetEnumerators::P2PXEC)) {
    // use structure-type <-> name translation for PhaseII geometry
    return Geometry::PhaseII;

  } else if (geometry->isThere(GeomDetEnumerators::P1PXEC)) {
    // use structure-type <-> name translation for PhaseI geometry
    return Geometry::PhaseI;

  } else if (geometry->isThere(GeomDetEnumerators::PixelEndcap)) {
    // use structure-type <-> name translation for RunI geometry
    return Geometry::RunI;

  } else {
    throw cms::Exception("AlignableObjectIdError") << "@SUB=AlignableObjectId::trackerGeometry\n"
                                                   << "unknown version of TrackerGeometry";
  }
}

AlignableObjectId::Geometry AlignableObjectId ::muonGeometry(const DTGeometry *,
                                                             const CSCGeometry *,
                                                             const GEMGeometry *) {
  // muon alignment structure types are identical for all kinds of geometries
  return Geometry::General;
}

AlignableObjectId::Geometry AlignableObjectId ::commonGeometry(Geometry first, Geometry second) {
  if (first == Geometry::General)
    return second;
  if (second == Geometry::General)
    return first;
  if (first == second)
    return first;

  throw cms::Exception("AlignableObjectIdError") << "@SUB=AlignableObjectId::commonGeometry\n"
                                                 << "impossible to find common geometry because the two geometries are "
                                                 << "different and none of them is 'General'";
}

AlignableObjectId AlignableObjectId ::commonObjectIdProvider(const AlignableObjectId &first,
                                                             const AlignableObjectId &second) {
  return AlignableObjectId{commonGeometry(first.geometry(), second.geometry())};
}
