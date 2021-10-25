#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "Geometry/MuonNumbering/interface/CSCNumberingScheme.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/MuonGeometryConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define EDM_ML_DEBUG

CSCNumberingScheme::CSCNumberingScheme(const MuonGeometryConstants& muonConstants) { initMe(muonConstants); }

void CSCNumberingScheme::initMe(const MuonGeometryConstants& muonConstants) {
  int theLevelPart = muonConstants.getValue("level");
  theRegionLevel = muonConstants.getValue("me_region") / theLevelPart;
  theStationLevel = muonConstants.getValue("me_station") / theLevelPart;
  theSubringLevel = muonConstants.getValue("me_subring") / theLevelPart;
  theSectorLevel = muonConstants.getValue("me_sector") / theLevelPart;
  theRingLevel = muonConstants.getValue("me_ring") / theLevelPart;
  theLayerLevel = muonConstants.getValue("me_layer") / theLevelPart;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "Initialize CSCNumberingScheme"
                               << "\ntheRegionLevel " << theRegionLevel << "\ntheStationLevel " << theStationLevel
                               << "\ntheSubringLevel " << theSubringLevel << "\ntheSectorLevel " << theSectorLevel
                               << "\ntheRingLevel " << theRingLevel << "\ntheLayerLevel " << theLayerLevel;
#endif
}

int CSCNumberingScheme::baseNumberToUnitNumber(const MuonBaseNumber& num) const {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "CSCNumbering " << num.getLevels();
  for (int level = 1; level <= num.getLevels(); level++) {
    edm::LogVerbatim("MuonGeom") << level << " " << num.getSuperNo(level) << " " << num.getBaseNo(level);
  }
#endif
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
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("MuonGeom") << "endcap=" << fwbw_id;
#endif
    } else if (level == theStationLevel) {
      const int station_tag = num.getSuperNo(level);
      station_id = station_tag;
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("MuonGeom") << "station=" << station_id;
#endif
    } else if (level == theSubringLevel) {
      const int copyno = num.getBaseNo(level);
      subring_id = copyno + 1;
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("MuonGeom") << "subring=" << subring_id;
#endif
    } else if (level == theSectorLevel) {
      const int copyno = num.getBaseNo(level);
      sector_id = copyno + 1;
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("MuonGeom") << "sector=" << sector_id;
#endif
    } else if (level == theLayerLevel) {
      const int copyno = num.getBaseNo(level);
      layer_id = copyno + 1;
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("MuonGeom") << "layer=" << layer_id;
#endif
    } else if (level == theRingLevel) {
      const int ring_tag = num.getSuperNo(level);
      ring_id = ring_tag;
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("MuonGeom") << "ring=" << ring_id;
#endif
    }
  }

  // check validity

  if ((fwbw_id < 1) || (fwbw_id > 2)) {
    edm::LogError("MuonGeom") << "@SUB=CSCNumberingScheme::baseNumberToUnitNumber"
                              << "forward/backward id out of range:" << fwbw_id;
  }

  if ((station_id < 1) || (station_id > 4)) {
    edm::LogError("MuonGeom") << "@SUB=CSCNumberingScheme::baseNumberToUnitNumber"
                              << "station id out of range:" << station_id;
  }

  if ((ring_id < 1) || (ring_id > 4)) {
    edm::LogError("MuonGeom") << "@SUB=CSCNumberingScheme::baseNumberToUnitNumber"
                              << "ring id out of range:" << ring_id;
  }

  if ((subring_id < 1) || (subring_id > 2)) {
    edm::LogError("MuonGeom") << "@SUB=CSCNumberingScheme::baseNumberToUnitNumber"
                              << "subring id out of range:" << subring_id;
  }

  if ((sector_id < 1) || (sector_id > 36)) {
    edm::LogError("MuonGeom") << "@SUB=CSCNumberingScheme::baseNumberToUnitNumber"
                              << "sector id out of range:" << sector_id;
  }

  // Allow id=0 since that means a chamber
  if ((layer_id < 0) || (layer_id > 6)) {
    edm::LogError("MuonGeom") << "@SUB=CSCNumberingScheme::baseNumberToUnitNumber"
                              << "layer id out of range" << layer_id;
  }

  // find appropriate chamber label

  int chamber_id = chamberIndex(station_id, ring_id, subring_id, sector_id);

  // convert into raw id of appropriate DetId

  int intIndex = CSCDetId::rawIdMaker(fwbw_id, station_id, ring_id, chamber_id, layer_id);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "CSCNumberingScheme :  fw/bw " << fwbw_id << " station " << station_id << " ring "
                               << ring_id << " subring " << subring_id << " chamber " << chamber_id << " sector "
                               << sector_id << " layer " << layer_id;
#endif

  return intIndex;
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
