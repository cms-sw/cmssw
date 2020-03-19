/*
// \class CSCNumberingScheme
//
//  Description: CSC Numbering Scheme for DD4hep
//              
//
// \author Sergio Lo Meo (sergio.lo.meo@cern.ch) following what Ianna Osburne made for DTs (DD4HEP migration)
//         Created:  Thu, 05 March 2020 
//   
//         Old DD version authors:  Arno Straessner & Tim Cox
*/
//
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "Geometry/MuonNumbering/interface/DD4hep_CSCNumberingScheme.h"
#include "Geometry/MuonNumbering/interface/DD4hep_MuonNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include <cassert>

using namespace cms;

CSCNumberingScheme::CSCNumberingScheme(const MuonConstants& muonConstants) { initMe(muonConstants); }

void CSCNumberingScheme::initMe(const MuonConstants& muonConstants) {
  int theLevelPart = get("level", muonConstants);
  assert(theLevelPart != 0);
  theRegionLevel = get("me_region", muonConstants) / theLevelPart;
  theStationLevel = get("me_station", muonConstants) / theLevelPart;
  theSubringLevel = get("me_subring", muonConstants) / theLevelPart;
  theSectorLevel = get("me_sector", muonConstants) / theLevelPart;
  theRingLevel = get("me_ring", muonConstants) / theLevelPart;
  theLayerLevel = get("me_layer", muonConstants) / theLevelPart;
}

void CSCNumberingScheme::baseNumberToUnitNumber(const MuonBaseNumber& num) {
  int fwbw_id = 0;
  int station_id = 0;
  int ring_id = 0;
  int subring_id = 0;
  int sector_id = 0;
  int layer_id = 0;

  // Decode endcap levels
  // We should be able to work with 6 (layer-level) or 5 (chamber-level)

  for (int level = 1; level <= num.getLevels(); level++) {
    if (level == theRegionLevel) {
      const int copyno = num.getBaseNo(level);
      fwbw_id = copyno + 1;
    } else if (level == theStationLevel) {
      const int station_tag = num.getSuperNo(level);
      station_id = station_tag;
    } else if (level == theSubringLevel) {
      const int copyno = num.getBaseNo(level);
      subring_id = copyno + 1;
    } else if (level == theSectorLevel) {
      const int copyno = num.getBaseNo(level);
      sector_id = copyno + 1;
    } else if (level == theLayerLevel) {
      const int copyno = num.getBaseNo(level);
      layer_id = copyno + 1;
    } else if (level == theRingLevel) {
      const int ring_tag = num.getSuperNo(level);
      ring_id = ring_tag;
    }
  }

  // find appropriate chamber label

  int chamber_id = chamberIndex(station_id, ring_id, subring_id, sector_id);

  // convert into raw id of appropriate DetId

  int intIndex = CSCDetId::rawIdMaker(fwbw_id, station_id, ring_id, chamber_id, layer_id);

  setDetId(intIndex);
}

int CSCNumberingScheme::chamberIndex(int station_id, int ring_id, int subring_id, int sector_id) const {
  int chamber_id = 0;

  // chamber label is related to sector_id but we need to
  // adjust to real hardware labelling
  // Tim confirms this works properly according to CMS IN 2000/004 Version 2.5 March 2007.

  if (ring_id == 3) {
    chamber_id = sector_id;
  } else {
    if (subring_id == 1) {
      chamber_id = 2 * sector_id - 1;
    } else {
      chamber_id = 2 * sector_id;
    }
  }

  return chamber_id;
}

const int CSCNumberingScheme::get(const char* key, const MuonConstants& muonConstants) const {
  int result(0);
  auto const& it = (muonConstants.find(key));
  if (it != end(muonConstants))
    result = it->second;
  return result;
}
