#include "Geometry/MuonNumbering/interface/ME0NumberingScheme.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/MuonGeometryConstants.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

//#define EDM_ML_DEBUG

ME0NumberingScheme::ME0NumberingScheme(const MuonGeometryConstants& muonConstants) { initMe(muonConstants); }

void ME0NumberingScheme::initMe(const MuonGeometryConstants& muonConstants) {
  int theLevelPart = muonConstants.getValue("level");
  theRegionLevel = muonConstants.getValue("m0_region") / theLevelPart;
  theLayerLevel = muonConstants.getValue("m0_layer") / theLevelPart;
  theSectorLevel = muonConstants.getValue("m0_sector") / theLevelPart;
  theRollLevel = muonConstants.getValue("m0_roll") / theLevelPart;
  theNEtaPart = muonConstants.getValue("m0_nroll");

  // Debug using EDM_ML_DEBUG
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "Initialize ME0NumberingScheme"
                               << "\ntheRegionLevel " << theRegionLevel << "\ntheLayerLevel " << theLayerLevel
                               << "\ntheSectorLevel " << theSectorLevel << "\ntheRollLevel " << theRollLevel
                               << "\ntheNEtaPart  " << theNEtaPart;
#endif
  // -----------------------
}

int ME0NumberingScheme::baseNumberToUnitNumber(const MuonBaseNumber& num) const {
  // Debug using EDM_ML_DEBUG
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "ME0NumberingScheme::baseNumberToUnitNumber BEGIN ";
  edm::LogVerbatim("MuonGeom") << "ME0Numbering " << num.getLevels();
  for (int level = 1; level <= num.getLevels(); level++) {
    edm::LogVerbatim("MuonGeom") << "level " << level << " " << num.getSuperNo(level) << " " << num.getBaseNo(level);
  }
#endif
  // -----------------------

#ifdef EDM_ML_DEBUG
  if (num.getLevels() != theRollLevel)
    edm::LogVerbatim("MuonGeom") << "MuonME0NS::BNToUN BaseNumber has " << num.getLevels()
                                 << " levels which is less than " << theRollLevel;
#endif

  int region(ME0DetId::minRegionId), layer(ME0DetId::minLayerId);
  int chamber(ME0DetId::minChamberId), roll(ME0DetId::minRollId);

  //decode significant ME0 levels

  if (num.getBaseNo(theRegionLevel) == 0)
    region = 1;
  else
    region = -1;
  if (num.getLevels() >= theLayerLevel)
    layer = num.getBaseNo(theLayerLevel) + 1;
  if (num.getLevels() >= theSectorLevel)
    chamber = num.getBaseNo(theSectorLevel) + 1;
  if (num.getLevels() >= theRollLevel)
    roll = num.getBaseNo(theRollLevel) + 1;

    // collect all info

    // Debug using EDM_ML_DEBUG
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "ME0NumberingScheme: Region " << region << " Layer " << layer << " Chamber "
                               << chamber << " Roll " << roll;
#endif
  // -----------------------

  // Build the actual numbering
  ME0DetId id(region, layer, chamber, roll);

  // Debug using EDM_ML_DEBUG
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << " DetId " << id;
#endif
  // ---------------------

  return id.rawId();
}
