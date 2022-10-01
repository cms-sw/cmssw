#include "CondFormats/RunInfo/interface/LHCInfoPerLS.h"
//#include "CondFormats/Common/interface/TimeConversions.h"
#include <algorithm>
#include <iterator>
#include <vector>
#include <stdexcept>

LHCInfoPerLS::LHCInfoPerLS() : LHCInfoVectorizedFields(ISIZE, FSIZE, TSIZE, SSIZE) {}

//getters
unsigned short const LHCInfoPerLS::fillNumber() const { return LHCInfoPerLS::getOneParam(m_intParams, LHC_FILL); }

unsigned int const& LHCInfoPerLS::lumiSection() const { return LHCInfoPerLS::getOneParam(m_intParams, LUMI_SECTION); }

float const LHCInfoPerLS::crossingAngleX() const { return LHCInfoPerLS::getOneParam(m_floatParams, CROSSING_ANGLE_X); }

float const LHCInfoPerLS::crossingAngleY() const { return LHCInfoPerLS::getOneParam(m_floatParams, CROSSING_ANGLE_Y); }

float const LHCInfoPerLS::betaStarX() const { return LHCInfoPerLS::getOneParam(m_floatParams, BETA_STAR_X); }

float const LHCInfoPerLS::betaStarY() const { return LHCInfoPerLS::getOneParam(m_floatParams, BETA_STAR_Y); }

cond::Time_t LHCInfoPerLS::runNumber() const { return LHCInfoPerLS::getOneParam(m_timeParams, RUN_NUMBER); }

//setters
void LHCInfoPerLS::setFillNumber(unsigned short const& lhcFill) {
  LHCInfoPerLS::setOneParam(m_intParams, LHC_FILL, static_cast<unsigned int>(lhcFill));
}

void LHCInfoPerLS::setLumiSection(unsigned int const& lumiSection) {
  LHCInfoPerLS::setOneParam(m_intParams, LUMI_SECTION, lumiSection);
}

void LHCInfoPerLS::setCrossingAngleX(float const& angle) {
  LHCInfoPerLS::setOneParam(m_floatParams, CROSSING_ANGLE_X, angle);
}

void LHCInfoPerLS::setCrossingAngleY(float const& angle) {
  LHCInfoPerLS::setOneParam(m_floatParams, CROSSING_ANGLE_Y, angle);
}

void LHCInfoPerLS::setBetaStarX(float const& betaStar) {
  LHCInfoPerLS::setOneParam(m_floatParams, BETA_STAR_X, betaStar);
}

void LHCInfoPerLS::setBetaStarY(float const& betaStar) {
  LHCInfoPerLS::setOneParam(m_floatParams, BETA_STAR_Y, betaStar);
}

void LHCInfoPerLS::setRunNumber(cond::Time_t const& runNumber) {
  LHCInfoPerLS::setOneParam(m_timeParams, RUN_NUMBER, runNumber);
};

//sets all values in one go
void LHCInfoPerLS::setInfo(unsigned short const& lhcFill,
                           unsigned int const& lumiSection,
                           float const& angleX,
                           float const& angleY,
                           float const& betaX,
                           float const& betaY,
                           cond::Time_t const& runNumber) {
  this->setFillNumber(lhcFill);
  this->setLumiSection(lumiSection);
  this->setCrossingAngleX(angleX);
  this->setCrossingAngleY(angleY);
  this->setBetaStarX(betaX);
  this->setBetaStarY(betaY);
  this->setRunNumber(runNumber);
}

void LHCInfoPerLS::print(std::stringstream& ss) const {
  ss << "LHC fill: " << this->fillNumber() << std::endl
     << "Lumi section: " << this->lumiSection() << std::endl
     << "Run number: " << this->runNumber() << std::endl
     << "Crossing angle x (urad): " << this->crossingAngleX() << std::endl
     << "Crossing angle y (urad): " << this->crossingAngleY() << std::endl
     << "Beta star x (m): " << this->betaStarX() << std::endl
     << "Beta star y (m): " << this->betaStarY() << std::endl;
}

std::ostream& operator<<(std::ostream& os, LHCInfoPerLS beamInfo) {
  std::stringstream ss;
  beamInfo.print(ss);
  os << ss.str();
  return os;
}

bool LHCInfoPerLS::equals(const LHCInfoPerLS& rhs) const {
  if (m_isData != rhs.m_isData)
    return false;
  if (m_intParams != rhs.m_intParams)
    return false;
  if (m_floatParams != rhs.m_floatParams)
    return false;
  if (m_timeParams != rhs.m_timeParams)
    return false;
  return true;
}

bool LHCInfoPerLS::empty() const { return m_intParams[0].empty(); }
