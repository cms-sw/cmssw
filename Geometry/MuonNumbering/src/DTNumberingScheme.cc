/** \class DTNumberingScheme
 *
 * implementation of MuonNumberingScheme for muon barrel,
 * converts the MuonBaseNumber to a unit id
 *  
 * Original \author Arno Straessner, CERN <arno.straessner@cern.ch>
 *         Modified by Sunanda B. in different PRs (the last one is #30971)
 *
 */

#include "Geometry/MuonNumbering/interface/DTNumberingScheme.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/MuonGeometryConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define EDM_ML_DEBUG

DTNumberingScheme::DTNumberingScheme(const MuonGeometryConstants& muonConstants) { initMe(muonConstants); }

void DTNumberingScheme::initMe(const MuonGeometryConstants& muonConstants) {
  int theLevelPart = muonConstants.getValue("level");
  theRegionLevel = muonConstants.getValue("mb_region") / theLevelPart;
  theWheelLevel = muonConstants.getValue("mb_wheel") / theLevelPart;
  theStationLevel = muonConstants.getValue("mb_station") / theLevelPart;
  theSuperLayerLevel = muonConstants.getValue("mb_superlayer") / theLevelPart;
  theLayerLevel = muonConstants.getValue("mb_layer") / theLevelPart;
  theWireLevel = muonConstants.getValue("mb_wire") / theLevelPart;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "Initialize DTNumberingScheme\ntheRegionLevel " << theRegionLevel
                               << "\ntheWheelLevel " << theWheelLevel << "\ntheStationLevel " << theStationLevel
                               << "\ntheSuperLayerLevel " << theSuperLayerLevel << "\ntheLayerLevel " << theLayerLevel
                               << "\ntheWireLevel " << theWireLevel;
#endif
}

int DTNumberingScheme::baseNumberToUnitNumber(const MuonBaseNumber& num) const {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "DTNumbering " << num.getLevels();
  for (int level = 1; level <= num.getLevels(); level++) {
    edm::LogVerbatim("MuonGeom") << level << " " << num.getSuperNo(level) << " " << num.getBaseNo(level);
  }
#endif
  if (num.getLevels() < theStationLevel) {  // it was  if (num.getLevels() != theWireLevel) {
    edm::LogWarning("MuonGeom") << "DTNumberingScheme::BNToUN: BaseNumber has " << num.getLevels() << " levels, need "
                                << theStationLevel;  //it was theWireLevel;
    return 0;
  }

  //   // Meaningful ranges are enforced by DTWireId, (which
  //   // however allows for 0 in wire, layer, superlayer!!!)
  //
  //   if ((wire_id < 1) || (wire_id > 100)) {
  //     edm::LogVerbatim("MuonNumbering") << "DTNumberingScheme: wire id out of range: " << wire_id;
  //   }

  //   if ((layer_id < 1) || (layer_id > 4)) {
  //     edm::LogVerbatim("MuonNumbering") << "DTNumberingScheme: layer id out of range: " << layer_id;
  //   }

  //   if ((superlayer_id < 1) || (superlayer_id > 3)) {
  //     edm::LogVerbatim("MuonNumbering") << "DTNumberingScheme: super-layer id out of range: " << superlayer_id;
  //   }

  return getDetId(num);
}

int DTNumberingScheme::getDetId(const MuonBaseNumber& num) const {
  int wire_id = 0;
  int layer_id = 0;
  int superlayer_id = 0;
  int sector_id = 0;
  int station_id = 0;
  int wheel_id = 0;

  //decode significant barrel levels
  decode(num, wire_id, layer_id, superlayer_id, sector_id, station_id, wheel_id);

  // These ranges are enforced by DTWireId
  //   if ((sector_id < 1) || (sector_id > 14)) {
  //     edm::LogVerbatim("MuonNumbering") << "DTNumberingScheme: sector id out of range: " << sector_id;
  //   }

  //   if ((station_id < 1) || (station_id > 4)) {
  //     edm::LogVerbatim("MuonNumbering") << "DTNumberingScheme: station id out of range: " << station_id;
  //   }

  //   if ((wheel_id < -2) || (wheel_id > 2)) {
  //     edm::LogVerbatim("MuonNumbering") << "DTNumberingScheme: wheel id out of range: " << wheel_id;
  //   }

  DTWireId id(wheel_id, station_id, sector_id, superlayer_id, layer_id, wire_id);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "DTNumberingScheme: " << id;
#endif

  return id.rawId();
}

void DTNumberingScheme::decode(const MuonBaseNumber& num,
                               int& wire_id,
                               int& layer_id,
                               int& superlayer_id,
                               int& sector_id,
                               int& station_id,
                               int& wheel_id) const {
  for (int level = 1; level <= num.getLevels(); level++) {
    //decode
    if (level == theWheelLevel) {
      const int copyno = num.getBaseNo(level);
      wheel_id = copyno - 2;

    } else if (level == theStationLevel) {
      const int station_tag = num.getSuperNo(level);
      const int copyno = num.getBaseNo(level);
      station_id = station_tag;
      sector_id = copyno + 1;

    } else if (level == theSuperLayerLevel) {
      const int copyno = num.getBaseNo(level);
      superlayer_id = copyno + 1;

    } else if (level == theLayerLevel) {
      const int copyno = num.getBaseNo(level);
      layer_id = copyno + 1;

    } else if (level == theWireLevel) {
      const int copyno = num.getBaseNo(level);
      wire_id = copyno + 1;
    }
  }
}
