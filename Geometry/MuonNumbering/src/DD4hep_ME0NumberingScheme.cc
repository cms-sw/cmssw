/*
//\class ME0NumberingScheme

Description: ME0 Numbering Scheme for DD4hep

//
// Author:  Sergio Lo Meo (sergio.lo.meo@cern.ch) following what Ianna Osburne made for DTs (DD4HEP migration)
//          Created:  29 Apr 2020 
*/
#include "Geometry/MuonNumbering/interface/DD4hep_ME0NumberingScheme.h"
#include "Geometry/MuonNumbering/interface/DD4hep_MuonNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <cassert>

using namespace cms;

ME0NumberingScheme::ME0NumberingScheme(const MuonConstants& muonConstants) { initMe(muonConstants); }

void ME0NumberingScheme::initMe(const MuonConstants& muonConstants) {
  int theLevelPart = get("level", muonConstants);

  assert(theLevelPart != 0);

  theRegionLevel = get("m0_region", muonConstants) / theLevelPart;
  theLayerLevel = get("m0_layer", muonConstants) / theLevelPart;
  theSectorLevel = get("m0_sector", muonConstants) / theLevelPart;
  theRollLevel = get("m0_roll", muonConstants) / theLevelPart;
  theNEtaPart = get("m0_nroll", muonConstants);
}

void ME0NumberingScheme::baseNumberToUnitNumber(const MuonBaseNumber& num) {
  int region = 0;
  int layer = 0;
  int chamber = 0;
  int roll = 0;

  //decode significant ME0 levels

  if (num.getBaseNo(theRegionLevel) == 0)
    region = 1;
  else
    region = -1;
  layer = num.getBaseNo(theLayerLevel) + 1;
  chamber = num.getBaseNo(theSectorLevel) + 1;
  roll = num.getBaseNo(theRollLevel) + 1;

  // Build the actual numbering
  ME0DetId id(region, layer, chamber, roll);

  setDetId(id.rawId());
}

const int ME0NumberingScheme::get(const char* key, const MuonConstants& muonConstants) const {
  int result(0);
  auto const& it = (muonConstants.find(key));
  if (it != end(muonConstants))
    result = it->second;
  return result;
}
