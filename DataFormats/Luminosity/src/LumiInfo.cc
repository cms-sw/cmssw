#include "DataFormats/Luminosity/interface/LumiInfo.h"

#include <iomanip>
#include <ostream>
#include <iostream>

float LumiInfo::instLuminosity() const {
  float totLum = 0;
  for (std::vector<float>::const_iterator it = instLumiByBX_.begin();
       it != instLumiByBX_.end(); ++it) {
    totLum += *it;
  }
  return totLum;
}

float LumiInfo::integLuminosity() const {
  return instLuminosity()*lumiSectionLength();
}

float LumiInfo::recordedLuminosity() const {
  return integLuminosity()*(1-deadtimeFraction_);
}

float LumiInfo::lumiSectionLength() const {
  // numorbits (262144)*numBX/orbit (3564)*24.95e-09
  return numOrbits_*numBX_*24.95e-9;
}

bool LumiInfo::isProductEqual(LumiInfo const& next) const {
  return (deadtimeFraction_ == next.deadtimeFraction_ &&
	  instLumiByBX_ == next.instLumiByBX_ &&
	  beam1Intensities_ == next.beam1Intensities_ &&
	  beam2Intensities_ == next.beam2Intensities_);
}

void LumiInfo::fill(const std::vector<float>& instLumiByBX,
		    const std::vector<float>& beam1Intensities,
		    const std::vector<float>& beam2Intensities) {
  instLumiByBX_.assign(instLumiByBX.begin(), instLumiByBX.end());
  beam1Intensities_.assign(beam1Intensities.begin(), beam1Intensities.end());
  beam2Intensities_.assign(beam2Intensities.begin(), beam2Intensities.end());
}

void LumiInfo::fillInstLumi(const std::vector<float>& instLumiByBX) {
  instLumiByBX_.assign(instLumiByBX.begin(), instLumiByBX.end());
}

void LumiInfo::fillBeamIntensities(const std::vector<float>& beam1Intensities,
				   const std::vector<float>& beam2Intensities) {
  beam1Intensities_.assign(beam1Intensities.begin(), beam1Intensities.end());
  beam2Intensities_.assign(beam2Intensities.begin(), beam2Intensities.end());
}

std::ostream& operator<<(std::ostream& s, const LumiInfo& lumiInfo) {
  s << "\nDumping LumiInfo\n\n";
  s << "  instLuminosity = " << lumiInfo.instLuminosity() << "\n";
  s << "  integLuminosity = " << lumiInfo.integLuminosity() << "\n";
  s << "  recordedLuminosity = " << lumiInfo.recordedLuminosity() << "\n";
  s << "  deadtimeFraction = " << lumiInfo.deadFraction() << "\n";
  s << "  instLumiByBX = ";
  const std::vector<float>& lumiBX = lumiInfo.getInstLumiAllBX();
  for (unsigned int i=0; i<10 && i<lumiBX.size(); ++i) {
    s << lumiBX.at(i) << " ";
  }
  s << " ...\n";
  s << "  beam1Intensities = ";
  const std::vector<float>& b1int = lumiInfo.getBeam1Intensities();
  for (unsigned int i=0; i<10 && i<b1int.size(); ++i) {
    s << b1int.at(i) << " ";
  }
  s << " ...\n";
  s << "  beam2Intensities = ";
  const std::vector<float>& b2int = lumiInfo.getBeam2Intensities();
  for (unsigned int i=0; i<10 && i<b2int.size(); ++i) {
    s << b2int.at(i) << " ";
  }
  s << " ...\n";

  return s << "\n";
}
