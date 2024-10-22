// -*- C++ -*-
//
// Package:     CalibFormats/SiStripObjects
// Class  :     SiStripDetInfo
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Fri, 28 May 2021 20:10:25 GMT
//

// system include files

// user include files
#include "CalibFormats/SiStripObjects/interface/SiStripDetInfo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

const std::pair<unsigned short, double> SiStripDetInfo::getNumberOfApvsAndStripLength(uint32_t detId) const {
  std::map<uint32_t, DetInfo>::const_iterator it = detData_.find(detId);

  if (it != detData_.end()) {
    return std::pair<unsigned short, double>(it->second.nApvs, it->second.stripLength);

  } else {
    std::pair<unsigned short, double> defaultValue(0, 0.);
    edm::LogWarning(
        "SiStripDetInfoFileReader::getNumberOfApvsAndStripLength - Unable to find requested detid. Returning invalid "
        "data ")
        << std::endl;
    return defaultValue;
  }
}

const float& SiStripDetInfo::getThickness(uint32_t detId) const {
  std::map<uint32_t, DetInfo>::const_iterator it = detData_.find(detId);

  if (it != detData_.end()) {
    return it->second.thickness;

  } else {
    static const float defaultValue = 0;
    edm::LogWarning("SiStripDetInfo::getThickness - Unable to find requested detid. Returning invalid data ")
        << std::endl;
    return defaultValue;
  }
}
