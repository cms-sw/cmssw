#include "DataFormats/Luminosity/interface/LumiInfo.h"

#include <vector>
#include <iomanip>
#include <ostream>
#include <iostream>

float LumiInfo::instLuminosityBXSum() const {
  float totLum = 0;
  for (std::vector<float>::const_iterator it = instLumiByBX_.begin();
       it != instLumiByBX_.end(); ++it) {
    totLum += *it;
  }
  return totLum;
}

float LumiInfo::integLuminosity() const {
  return getTotalInstLumi()*lumiSectionLength();
}

void LumiInfo::setTotalInstToBXSum() {
  setTotalInstLumi(instLuminosityBXSum());
}

float LumiInfo::recordedLuminosity() const {
  return integLuminosity()*(1-deadtimeFraction_);
}

float LumiInfo::lumiSectionLength() const {
  // numorbits (262144)*numBX/orbit (3564)*bx spacing (24.95e-09)
  return LumiConstants::numOrbits*LumiConstants::numBX*LumiConstants::bxSpacingExact;
}

bool LumiInfo::isProductEqual(LumiInfo const& next) const {
  return (deadtimeFraction_ == next.deadtimeFraction_ &&
	  instLumiByBX_ == next.instLumiByBX_);
}

void LumiInfo::setInstLumiAllBX(std::vector<float>& instLumiByBX) {
  instLumiByBX_.assign(instLumiByBX.begin(), instLumiByBX.end());
}

void LumiInfo::setErrorLumiAllBX(std::vector<float>& errLumiByBX){
  instLumiStatErrByBX_.assign(errLumiByBX.begin(),errLumiByBX.end());
}

std::ostream& operator<<(std::ostream& s, const LumiInfo& lumiInfo) {
  s << "\nDumping LumiInfo\n\n";
  s << "  getTotalInstLumi = " << lumiInfo.getTotalInstLumi() << "\n";
  s << "  integLuminosity = " << lumiInfo.integLuminosity() << "\n";
  s << "  recordedLuminosity = " << lumiInfo.recordedLuminosity() << "\n";
  s << "  deadtimeFraction = " << lumiInfo.getDeadFraction() << "\n";
  s << "  instLumiByBX = ";
  const std::vector<float>& lumiBX = lumiInfo.getInstLumiAllBX();
  for (unsigned int i=0; i<10 && i<lumiBX.size(); ++i) {
    s << lumiBX.at(i) << " ";
  }
  s << " ...\n";

  return s << "\n";
}
