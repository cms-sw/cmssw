/*
//\class GEMNumberingScheme

Description: GEM Numbering Scheme for DD4HEP

//
// Author:  Sergio Lo Meo (sergio.lo.meo@cern.ch) following what Ianna Osburne made for DTs (DD4HEP migration)
//          Created:  Mon, 27 Jan 2020 
*/
#include "Geometry/MuonNumbering/interface/DD4hep_GEMNumberingScheme.h"
#include "Geometry/MuonNumbering/interface/DD4hep_MuonNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <cassert>

using namespace cms;

GEMNumberingScheme::GEMNumberingScheme(const MuonConstants& muonConstants) { initMe(muonConstants); }

void GEMNumberingScheme::initMe(const MuonConstants& muonConstants) {
  int levelPart = get("level", muonConstants);

  assert(levelPart != 0);
  theRegionLevel = get("mg_region", muonConstants) / levelPart;
  theStationLevel = get("mg_station", muonConstants) / levelPart;
  theRingLevel = get("mg_ring", muonConstants) / levelPart;
  theSectorLevel = get("mg_sector", muonConstants) / levelPart;
  theRollLevel = get("mg_roll", muonConstants) / levelPart;
}

void GEMNumberingScheme::baseNumberToUnitNumber(const MuonBaseNumber& num) {
  int maxLevel = theRollLevel;
  if (num.getLevels() != maxLevel) {
    edm::LogWarning("GEMNumberingScheme")
        << "MuonGEMNumberingScheme::BNToUN: BaseNumber has " << num.getLevels() << " levels, need " << maxLevel;
  }

  int region = 0;
  int ring = 0;
  int station = 0;
  int layer = 0;
  int chamber = 0;
  int roll = 0;

  //decode significant GEM levels

  if (num.getBaseNo(theRegionLevel) == 0)
    region = 1;
  else
    region = -1;

  // All GEM super chambers in stations 1 and 2 are on ring 1.
  // The long super chambers in station 2 are assigned *station 3* due
  // to the current limitation in the definition of the GEMDetId,
  // i.e. only 2 layers available per station.
  //  ring    = num.getSuperNo(theRingLevel);
  // GEM are only on the first ring
  ring = 1;
  station = num.getSuperNo(theStationLevel);

  roll = num.getBaseNo(theRollLevel) + 1;
  const int copyno = num.getBaseNo(theSectorLevel) + 1;
  const int maxcno = 50;
  if (copyno < maxcno) {
    if (copyno % 2 == 0) {
      layer = 2;
      chamber = copyno - 1;
    } else {
      layer = 1;
      chamber = copyno;
    }
  } else {
    int copynp = copyno - maxcno;
    if (copynp % 2 != 0) {
      layer = 2;
      chamber = copynp - 1;
    } else {
      layer = 1;
      chamber = copynp;
    }
  }

  // Build the actual numbering
  GEMDetId id(region, ring, station, layer, chamber, roll);

  setDetId(id.rawId());
}

const int GEMNumberingScheme::get(const char* key, const MuonConstants& muonConstants) const {
  int result(0);
  auto const& it = (muonConstants.find(key));
  if (it != end(muonConstants))
    result = it->second;
  return result;
}
