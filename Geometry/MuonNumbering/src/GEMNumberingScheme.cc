#include "Geometry/MuonNumbering/interface/GEMNumberingScheme.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/MuonGeometryConstants.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define EDM_ML_DEBUG

GEMNumberingScheme::GEMNumberingScheme(const MuonGeometryConstants& muonConstants) { initMe(muonConstants); }

void GEMNumberingScheme::initMe(const MuonGeometryConstants& muonConstants) {
  int theLevelPart = muonConstants.getValue("level");
  theRegionLevel = muonConstants.getValue("mg_region") / theLevelPart;
  theStationLevel = muonConstants.getValue("mg_station") / theLevelPart;
  theRingLevel = muonConstants.getValue("mg_ring") / theLevelPart;
  theSectorLevel = muonConstants.getValue("mg_sector") / theLevelPart;
  theRollLevel = muonConstants.getValue("mg_roll") / theLevelPart;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "Initialize GEMNumberingScheme"
                               << "\ntheRegionLevel " << theRegionLevel << "\ntheStationLevel " << theStationLevel
                               << "\ntheRingLevel " << theRingLevel << "\ntheSectorLevel " << theSectorLevel
                               << "\ntheRollLevel " << theRollLevel;
#endif
}

int GEMNumberingScheme::baseNumberToUnitNumber(const MuonBaseNumber& num) const {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "GEMNumbering " << num.getLevels();
  for (int level = 1; level <= num.getLevels(); level++) {
    edm::LogVerbatim("MuonGeom") << level << " " << num.getSuperNo(level) << " " << num.getBaseNo(level);
  }
#endif

  int levels = num.getLevels();
#ifdef EDM_ML_DEBUG
  if (levels != theRollLevel)
    edm::LogVerbatim("MuonGeom") << "MuonGEMNumberingScheme::BNToUN: BaseNumber has " << num.getLevels()
                                 << " levels, need at least till " << theRollLevel;
#endif

  int region(GEMDetId::minRegionId), ring(GEMDetId::minRingId);
  int station(GEMDetId::minStationId0), layer(GEMDetId::minLayerId);
  int chamber(1 + GEMDetId::minChamberId), roll(GEMDetId::minRollId);

  //decode significant GEM levels

  if (levels >= theRegionLevel) {
    if (num.getBaseNo(theRegionLevel) == 0)
      region = 1;
    else
      region = -1;
  }

  // All GEM super chambers in stations 1 and 2 are on ring 1.
  // The long super chambers in station 2 are assigned *station 3* due
  // to the current limitation in the definition of the GEMDetId,
  // i.e. only 2 layers available per station.
  //  ring    = num.getSuperNo(theRingLevel);
  // GEM are only on the first ring
  ring = 1;

  // GE0 has the layer encoded in the ring level
  if (levels > theRingLevel) {
    if (num.getBaseNo(theRingLevel) == 0) {  // 0 => GE1/1, GE2/1
      station = num.getSuperNo(theStationLevel);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("MuonGeom") << "GEMNumbering: Ring " << ring << " Station " << num.getSuperNo(theStationLevel)
                                   << ":" << station;
#endif
      if (levels >= theRollLevel)
        roll = num.getBaseNo(theRollLevel) + 1;
      if (levels >= theSectorLevel) {
        const int copyno = num.getBaseNo(theSectorLevel) + 1;
        // Half the chambers are flipped back to front, this is encoded in
        // the chamber number, which affects the layer numbering. Layer 1
        // is always the closest layer to the interaction point.
        const int layerDemarcation = 50;
        if (copyno < layerDemarcation) {
          if (copyno % 2 == 0) {
            layer = 2;
            chamber = copyno - 1;
          } else {
            layer = 1;
            chamber = copyno;
          }
        } else {
          int copynp = copyno - layerDemarcation;
          if (copynp % 2 != 0) {
            layer = 2;
            chamber = copynp - 1;
          } else {
            layer = 1;
            chamber = copynp;
          }
        }
      }
    } else {  // GE0 encodes the layer
      station = GEMDetId::minStationId0;
      layer = num.getBaseNo(theRingLevel);
      if (levels >= theSectorLevel)
        chamber = num.getBaseNo(theSectorLevel) + 1;
      if (levels >= theRollLevel)
        roll = num.getBaseNo(theRollLevel) + 1;
    }
  } else if (levels == theRingLevel) {
    station = GEMDetId::minStationId0;
    layer = 1;
  }

  // collect all info

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "GEMNumberingScheme: Region " << region << " Ring " << ring << " Station " << station
                               << " Layer " << layer << " Chamber " << chamber << " Roll " << roll;
#endif

  // Build the actual numbering
  GEMDetId id(region, ring, station, layer, chamber, roll);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << id.rawId() << " DetId " << id;
#endif

  return id.rawId();
}
