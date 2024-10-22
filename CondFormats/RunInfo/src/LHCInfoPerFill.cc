#include "CondFormats/RunInfo/interface/LHCInfoPerFill.h"
#include "CondFormats/Common/interface/TimeConversions.h"
#include <algorithm>
#include <iterator>
#include <vector>
#include <stdexcept>

//helper function: returns the positions of the bits in the bitset that are set (i.e., have a value of 1).
static std::vector<unsigned short> bitsetToVector(std::bitset<LHCInfoPerFill::bunchSlots + 1> const& bs) {
  std::vector<unsigned short> vec;
  //reserve space only for the bits in the bitset that are set
  vec.reserve(bs.count());
  for (size_t i = 0; i < bs.size(); ++i) {
    if (bs.test(i))
      vec.push_back((unsigned short)i);
  }
  return vec;
}

//helper function: returns the enum for fill types in string type
static std::string fillTypeToString(LHCInfoPerFill::FillTypeId const& fillType) {
  std::string s_fillType("UNKNOWN");
  switch (fillType) {
    case LHCInfoPerFill::UNKNOWN:
      s_fillType = std::string("UNKNOWN");
      break;
    case LHCInfoPerFill::PROTONS:
      s_fillType = std::string("PROTONS");
      break;
    case LHCInfoPerFill::IONS:
      s_fillType = std::string("IONS");
      break;
    case LHCInfoPerFill::COSMICS:
      s_fillType = std::string("COSMICS");
      break;
    case LHCInfoPerFill::GAP:
      s_fillType = std::string("GAP");
      break;
    default:
      s_fillType = std::string("UNKNOWN");
  }
  return s_fillType;
}

//helper function: returns the enum for particle types in string type
static std::string particleTypeToString(LHCInfoPerFill::ParticleTypeId const& particleType) {
  std::string s_particleType("NONE");
  switch (particleType) {
    case LHCInfoPerFill::NONE:
      s_particleType = std::string("NONE");
      break;
    case LHCInfoPerFill::PROTON:
      s_particleType = std::string("PROTON");
      break;
    case LHCInfoPerFill::PB82:
      s_particleType = std::string("PB82");
      break;
    case LHCInfoPerFill::AR18:
      s_particleType = std::string("AR18");
      break;
    case LHCInfoPerFill::D:
      s_particleType = std::string("D");
      break;
    case LHCInfoPerFill::XE54:
      s_particleType = std::string("XE54");
      break;
    default:
      s_particleType = std::string("NONE");
  }
  return s_particleType;
}

LHCInfoPerFill::LHCInfoPerFill() : LHCInfoVectorizedFields(ISIZE, FSIZE, TSIZE, SSIZE) {
  m_floatParams[LUMI_PER_B] = std::vector<float>();
  m_floatParams[BEAM1_VC] = std::vector<float>();
  m_floatParams[BEAM2_VC] = std::vector<float>();
  m_floatParams[BEAM1_RF] = std::vector<float>();
  m_floatParams[BEAM2_RF] = std::vector<float>();
  m_stringParams[INJECTION_SCHEME].push_back(std::string("None"));
}

LHCInfoPerFill* LHCInfoPerFill::cloneFill() const {
  LHCInfoPerFill* ret = new LHCInfoPerFill();
  ret->m_isData = m_isData;
  if (!m_intParams[0].empty()) {
    for (size_t i = 0; i < ISIZE; i++)
      ret->m_intParams[i] = m_intParams[i];
    for (size_t i = 0; i < DELIV_LUMI; i++)
      ret->m_floatParams[i] = m_floatParams[i];
    ret->m_floatParams[LUMI_PER_B] = m_floatParams[LUMI_PER_B];
    for (size_t i = 0; i < TSIZE; i++)
      ret->m_timeParams[i] = m_timeParams[i];
    for (size_t i = 0; i < LHC_STATE; i++)
      ret->m_stringParams[i] = m_stringParams[i];
    ret->m_bunchConfiguration1 = m_bunchConfiguration1;
    ret->m_bunchConfiguration2 = m_bunchConfiguration2;
  }
  return ret;
}

//getters
unsigned short const LHCInfoPerFill::fillNumber() const { return LHCInfoPerFill::getOneParam(m_intParams, LHC_FILL); }

unsigned short const LHCInfoPerFill::bunchesInBeam1() const {
  return LHCInfoPerFill::getOneParam(m_intParams, BUNCHES_1);
}

unsigned short const LHCInfoPerFill::bunchesInBeam2() const {
  return LHCInfoPerFill::getOneParam(m_intParams, BUNCHES_2);
}

unsigned short const LHCInfoPerFill::collidingBunches() const {
  return LHCInfoPerFill::getOneParam(m_intParams, COLLIDING_BUNCHES);
}

unsigned short const LHCInfoPerFill::targetBunches() const {
  return LHCInfoPerFill::getOneParam(m_intParams, TARGET_BUNCHES);
}

LHCInfoPerFill::FillTypeId const LHCInfoPerFill::fillType() const {
  return static_cast<FillTypeId>(LHCInfoPerFill::getOneParam(m_intParams, FILL_TYPE));
}

LHCInfoPerFill::ParticleTypeId const LHCInfoPerFill::particleTypeForBeam1() const {
  return static_cast<ParticleTypeId>(LHCInfoPerFill::getOneParam(m_intParams, PARTICLES_1));
}

LHCInfoPerFill::ParticleTypeId const LHCInfoPerFill::particleTypeForBeam2() const {
  return static_cast<ParticleTypeId>(LHCInfoPerFill::getOneParam(m_intParams, PARTICLES_2));
}

float const LHCInfoPerFill::intensityForBeam1() const {
  return LHCInfoPerFill::getOneParam(m_floatParams, INTENSITY_1);
}

float const LHCInfoPerFill::intensityForBeam2() const {
  return LHCInfoPerFill::getOneParam(m_floatParams, INTENSITY_2);
}

float const LHCInfoPerFill::energy() const { return LHCInfoPerFill::getOneParam(m_floatParams, ENERGY); }

float const LHCInfoPerFill::delivLumi() const { return LHCInfoPerFill::getOneParam(m_floatParams, DELIV_LUMI); }

float const LHCInfoPerFill::recLumi() const { return LHCInfoPerFill::getOneParam(m_floatParams, REC_LUMI); }

float const LHCInfoPerFill::instLumi() const { return LHCInfoPerFill::getOneParam(m_floatParams, INST_LUMI); }

float const LHCInfoPerFill::instLumiError() const { return LHCInfoPerFill::getOneParam(m_floatParams, INST_LUMI_ERR); }

cond::Time_t const LHCInfoPerFill::createTime() const { return LHCInfoPerFill::getOneParam(m_timeParams, CREATE_TIME); }

cond::Time_t const LHCInfoPerFill::beginTime() const { return LHCInfoPerFill::getOneParam(m_timeParams, BEGIN_TIME); }

cond::Time_t const LHCInfoPerFill::endTime() const { return LHCInfoPerFill::getOneParam(m_timeParams, END_TIME); }

std::string const& LHCInfoPerFill::injectionScheme() const {
  return LHCInfoPerFill::getOneParam(m_stringParams, INJECTION_SCHEME);
}

std::vector<float> const& LHCInfoPerFill::lumiPerBX() const {
  return LHCInfoPerFill::getParams(m_floatParams, LUMI_PER_B);
}

std::string const& LHCInfoPerFill::lhcState() const { return LHCInfoPerFill::getOneParam(m_stringParams, LHC_STATE); }

std::string const& LHCInfoPerFill::lhcComment() const {
  return LHCInfoPerFill::getOneParam(m_stringParams, LHC_COMMENT);
}

std::string const& LHCInfoPerFill::ctppsStatus() const {
  return LHCInfoPerFill::getOneParam(m_stringParams, CTPPS_STATUS);
}

std::vector<float> const& LHCInfoPerFill::beam1VC() const { return LHCInfoPerFill::getParams(m_floatParams, BEAM1_VC); }

std::vector<float> const& LHCInfoPerFill::beam2VC() const { return LHCInfoPerFill::getParams(m_floatParams, BEAM2_VC); }

std::vector<float> const& LHCInfoPerFill::beam1RF() const { return LHCInfoPerFill::getParams(m_floatParams, BEAM1_RF); }

std::vector<float> const& LHCInfoPerFill::beam2RF() const { return LHCInfoPerFill::getParams(m_floatParams, BEAM2_RF); }

std::vector<float>& LHCInfoPerFill::beam1VC() { return LHCInfoPerFill::accessParams(m_floatParams, BEAM1_VC); }

std::vector<float>& LHCInfoPerFill::beam2VC() { return LHCInfoPerFill::accessParams(m_floatParams, BEAM2_VC); }

std::vector<float>& LHCInfoPerFill::beam1RF() { return LHCInfoPerFill::accessParams(m_floatParams, BEAM1_RF); }

std::vector<float>& LHCInfoPerFill::beam2RF() { return LHCInfoPerFill::accessParams(m_floatParams, BEAM2_RF); }

//returns a boolean, true if the injection scheme has a leading 25ns
//TODO: parse the circulating bunch configuration, instead of the string.
bool LHCInfoPerFill::is25nsBunchSpacing() const {
  const std::string prefix("25ns");
  return std::equal(prefix.begin(), prefix.end(), injectionScheme().begin());
}

//returns a boolean, true if the bunch slot number is in the circulating bunch configuration
bool LHCInfoPerFill::isBunchInBeam1(size_t const& bunch) const {
  if (bunch == 0)
    throw std::out_of_range("0 not allowed");  //CMS starts counting bunch crossing from 1!
  return m_bunchConfiguration1.test(bunch);
}

bool LHCInfoPerFill::isBunchInBeam2(size_t const& bunch) const {
  if (bunch == 0)
    throw std::out_of_range("0 not allowed");  //CMS starts counting bunch crossing from 1!
  return m_bunchConfiguration2.test(bunch);
}

//member functions returning *by value* a vector with all filled bunch slots
std::vector<unsigned short> LHCInfoPerFill::bunchConfigurationForBeam1() const {
  return bitsetToVector(m_bunchConfiguration1);
}

std::vector<unsigned short> LHCInfoPerFill::bunchConfigurationForBeam2() const {
  return bitsetToVector(m_bunchConfiguration2);
}

void LHCInfoPerFill::setFillNumber(unsigned short lhcFill) {
  LHCInfoPerFill::setOneParam(m_intParams, LHC_FILL, static_cast<unsigned int>(lhcFill));
}

//setters
void LHCInfoPerFill::setBunchesInBeam1(unsigned short const& bunches) {
  LHCInfoPerFill::setOneParam(m_intParams, BUNCHES_1, static_cast<unsigned int>(bunches));
}

void LHCInfoPerFill::setBunchesInBeam2(unsigned short const& bunches) {
  LHCInfoPerFill::setOneParam(m_intParams, BUNCHES_2, static_cast<unsigned int>(bunches));
}

void LHCInfoPerFill::setCollidingBunches(unsigned short const& collidingBunches) {
  LHCInfoPerFill::setOneParam(m_intParams, COLLIDING_BUNCHES, static_cast<unsigned int>(collidingBunches));
}

void LHCInfoPerFill::setTargetBunches(unsigned short const& targetBunches) {
  LHCInfoPerFill::setOneParam(m_intParams, TARGET_BUNCHES, static_cast<unsigned int>(targetBunches));
}

void LHCInfoPerFill::setFillType(LHCInfoPerFill::FillTypeId const& fillType) {
  LHCInfoPerFill::setOneParam(m_intParams, FILL_TYPE, static_cast<unsigned int>(fillType));
}

void LHCInfoPerFill::setParticleTypeForBeam1(LHCInfoPerFill::ParticleTypeId const& particleType) {
  LHCInfoPerFill::setOneParam(m_intParams, PARTICLES_1, static_cast<unsigned int>(particleType));
}

void LHCInfoPerFill::setParticleTypeForBeam2(LHCInfoPerFill::ParticleTypeId const& particleType) {
  LHCInfoPerFill::setOneParam(m_intParams, PARTICLES_2, static_cast<unsigned int>(particleType));
}

void LHCInfoPerFill::setIntensityForBeam1(float const& intensity) {
  LHCInfoPerFill::setOneParam(m_floatParams, INTENSITY_1, intensity);
}

void LHCInfoPerFill::setIntensityForBeam2(float const& intensity) {
  LHCInfoPerFill::setOneParam(m_floatParams, INTENSITY_2, intensity);
}

void LHCInfoPerFill::setEnergy(float const& energy) { LHCInfoPerFill::setOneParam(m_floatParams, ENERGY, energy); }

void LHCInfoPerFill::setDelivLumi(float const& delivLumi) {
  LHCInfoPerFill::setOneParam(m_floatParams, DELIV_LUMI, delivLumi);
}

void LHCInfoPerFill::setRecLumi(float const& recLumi) { LHCInfoPerFill::setOneParam(m_floatParams, REC_LUMI, recLumi); }

void LHCInfoPerFill::setInstLumi(float const& instLumi) {
  LHCInfoPerFill::setOneParam(m_floatParams, INST_LUMI, instLumi);
}

void LHCInfoPerFill::setInstLumiError(float const& instLumiError) {
  LHCInfoPerFill::setOneParam(m_floatParams, INST_LUMI_ERR, instLumiError);
}

void LHCInfoPerFill::setCreationTime(cond::Time_t const& createTime) {
  LHCInfoPerFill::setOneParam(m_timeParams, CREATE_TIME, createTime);
}

void LHCInfoPerFill::setBeginTime(cond::Time_t const& beginTime) {
  LHCInfoPerFill::setOneParam(m_timeParams, BEGIN_TIME, beginTime);
}

void LHCInfoPerFill::setEndTime(cond::Time_t const& endTime) {
  LHCInfoPerFill::setOneParam(m_timeParams, END_TIME, endTime);
}

void LHCInfoPerFill::setInjectionScheme(std::string const& injectionScheme) {
  LHCInfoPerFill::setOneParam(m_stringParams, INJECTION_SCHEME, injectionScheme);
}

void LHCInfoPerFill::setLumiPerBX(std::vector<float> const& lumiPerBX) {
  LHCInfoPerFill::setParams(m_floatParams, LUMI_PER_B, lumiPerBX);
}

void LHCInfoPerFill::setLhcState(std::string const& lhcState) {
  LHCInfoPerFill::setOneParam(m_stringParams, LHC_STATE, lhcState);
}

void LHCInfoPerFill::setLhcComment(std::string const& lhcComment) {
  LHCInfoPerFill::setOneParam(m_stringParams, LHC_COMMENT, lhcComment);
}

void LHCInfoPerFill::setCtppsStatus(std::string const& ctppsStatus) {
  LHCInfoPerFill::setOneParam(m_stringParams, CTPPS_STATUS, ctppsStatus);
}

void LHCInfoPerFill::setBeam1VC(std::vector<float> const& beam1VC) {
  LHCInfoPerFill::setParams(m_floatParams, BEAM1_VC, beam1VC);
}

void LHCInfoPerFill::setBeam2VC(std::vector<float> const& beam2VC) {
  LHCInfoPerFill::setParams(m_floatParams, BEAM2_VC, beam2VC);
}

void LHCInfoPerFill::setBeam1RF(std::vector<float> const& beam1RF) {
  LHCInfoPerFill::setParams(m_floatParams, BEAM1_RF, beam1RF);
}

void LHCInfoPerFill::setBeam2RF(std::vector<float> const& beam2RF) {
  LHCInfoPerFill::setParams(m_floatParams, BEAM2_RF, beam2RF);
}

//sets all values in one go
void LHCInfoPerFill::setInfo(unsigned short const& bunches1,
                             unsigned short const& bunches2,
                             unsigned short const& collidingBunches,
                             unsigned short const& targetBunches,
                             FillTypeId const& fillType,
                             ParticleTypeId const& particleType1,
                             ParticleTypeId const& particleType2,
                             float const& intensity1,
                             float const& intensity2,
                             float const& energy,
                             float const& delivLumi,
                             float const& recLumi,
                             float const& instLumi,
                             float const& instLumiError,
                             cond::Time_t const& createTime,
                             cond::Time_t const& beginTime,
                             cond::Time_t const& endTime,
                             std::string const& scheme,
                             std::vector<float> const& lumiPerBX,
                             std::string const& lhcState,
                             std::string const& lhcComment,
                             std::string const& ctppsStatus,
                             std::vector<float> const& beam1VC,
                             std::vector<float> const& beam2VC,
                             std::vector<float> const& beam1RF,
                             std::vector<float> const& beam2RF,
                             std::bitset<bunchSlots + 1> const& bunchConf1,
                             std::bitset<bunchSlots + 1> const& bunchConf2) {
  this->setBunchesInBeam1(bunches1);
  this->setBunchesInBeam2(bunches2);
  this->setCollidingBunches(collidingBunches);
  this->setTargetBunches(targetBunches);
  this->setFillType(fillType);
  this->setParticleTypeForBeam1(particleType1);
  this->setParticleTypeForBeam2(particleType2);
  this->setIntensityForBeam1(intensity1);
  this->setIntensityForBeam2(intensity2);
  this->setEnergy(energy);
  this->setDelivLumi(delivLumi);
  this->setRecLumi(recLumi);
  this->setInstLumi(instLumi);
  this->setInstLumiError(instLumiError);
  this->setCreationTime(createTime);
  this->setBeginTime(beginTime);
  this->setEndTime(endTime);
  this->setInjectionScheme(scheme);
  this->setLumiPerBX(lumiPerBX);
  this->setLhcState(lhcState);
  this->setLhcComment(lhcComment);
  this->setCtppsStatus(ctppsStatus);
  this->setBeam1VC(beam1VC);
  this->setBeam2VC(beam2VC);
  this->setBeam1RF(beam1RF);
  this->setBeam2RF(beam2RF);
  this->setBunchBitsetForBeam1(bunchConf1);
  this->setBunchBitsetForBeam2(bunchConf2);
}

void LHCInfoPerFill::print(std::stringstream& ss) const {
  ss << "LHC fill: " << this->fillNumber() << std::endl
     << "Bunches in Beam 1: " << this->bunchesInBeam1() << std::endl
     << "Bunches in Beam 2: " << this->bunchesInBeam2() << std::endl
     << "Colliding bunches at IP5: " << this->collidingBunches() << std::endl
     << "Target bunches at IP5: " << this->targetBunches() << std::endl
     << "Fill type: " << fillTypeToString(static_cast<FillTypeId>(this->fillType())) << std::endl
     << "Particle type for Beam 1: " << particleTypeToString(static_cast<ParticleTypeId>(this->particleTypeForBeam1()))
     << std::endl
     << "Particle type for Beam 2: " << particleTypeToString(static_cast<ParticleTypeId>(this->particleTypeForBeam2()))
     << std::endl
     << "Average Intensity for Beam 1 (number of charges): " << this->intensityForBeam1() << std::endl
     << "Average Intensity for Beam 2 (number of charges): " << this->intensityForBeam2() << std::endl
     << "Energy (GeV): " << this->energy() << std::endl
     << "Delivered Luminosity (max): " << this->delivLumi() << std::endl
     << "Recorded Luminosity (max): " << this->recLumi() << std::endl
     << "Instantaneous Luminosity: " << this->instLumi() << std::endl
     << "Instantaneous Luminosity Error: " << this->instLumiError() << std::endl
     << "Creation time of the fill: "
     << boost::posix_time::to_iso_extended_string(cond::time::to_boost(this->createTime())) << std::endl
     << "Begin time of Stable Beam flag: "
     << boost::posix_time::to_iso_extended_string(cond::time::to_boost(this->beginTime())) << std::endl
     << "End time of the fill: " << boost::posix_time::to_iso_extended_string(cond::time::to_boost(this->endTime()))
     << std::endl
     << "Injection scheme as given by LPC: " << this->injectionScheme() << std::endl
     << "LHC State: " << this->lhcState() << std::endl
     << "LHC Comments: " << this->lhcComment() << std::endl
     << "CTPPS Status: " << this->ctppsStatus() << std::endl;

  ss << "Luminosity per bunch  (total " << this->lumiPerBX().size() << "): ";
  std::copy(this->lumiPerBX().begin(), this->lumiPerBX().end(), std::ostream_iterator<float>(ss, ", "));
  ss << std::endl;

  ss << "Beam 1 VC  (total " << this->beam1VC().size() << "): ";
  std::copy(this->beam1VC().begin(), this->beam1VC().end(), std::ostream_iterator<float>(ss, "\t"));
  ss << std::endl;

  ss << "Beam 2 VC  (total " << beam2VC().size() << "): ";
  std::copy(beam2VC().begin(), beam2VC().end(), std::ostream_iterator<float>(ss, "\t"));
  ss << std::endl;

  ss << "Beam 1 RF  (total " << beam1RF().size() << "): ";
  std::copy(beam1RF().begin(), beam1RF().end(), std::ostream_iterator<float>(ss, "\t"));
  ss << std::endl;

  ss << "Beam 2 RF  (total " << beam2RF().size() << "): ";
  std::copy(beam2RF().begin(), beam2RF().end(), std::ostream_iterator<float>(ss, "\t"));
  ss << std::endl;

  std::vector<unsigned short> bunchVector1 = this->bunchConfigurationForBeam1();
  std::vector<unsigned short> bunchVector2 = this->bunchConfigurationForBeam2();
  ss << "Bunches filled for Beam 1 (total " << bunchVector1.size() << "): ";
  std::copy(bunchVector1.begin(), bunchVector1.end(), std::ostream_iterator<unsigned short>(ss, ", "));
  ss << std::endl;
  ss << "Bunches filled for Beam 2 (total " << bunchVector2.size() << "): ";
  std::copy(bunchVector2.begin(), bunchVector2.end(), std::ostream_iterator<unsigned short>(ss, ", "));
  ss << std::endl;
}

//protected getters
std::bitset<LHCInfoPerFill::bunchSlots + 1> const& LHCInfoPerFill::bunchBitsetForBeam1() const {
  return m_bunchConfiguration1;
}

std::bitset<LHCInfoPerFill::bunchSlots + 1> const& LHCInfoPerFill::bunchBitsetForBeam2() const {
  return m_bunchConfiguration2;
}

//protected setters
void LHCInfoPerFill::setBunchBitsetForBeam1(std::bitset<LHCInfoPerFill::bunchSlots + 1> const& bunchConfiguration) {
  m_bunchConfiguration1 = bunchConfiguration;
}

void LHCInfoPerFill::setBunchBitsetForBeam2(std::bitset<LHCInfoPerFill::bunchSlots + 1> const& bunchConfiguration) {
  m_bunchConfiguration2 = bunchConfiguration;
}

std::ostream& operator<<(std::ostream& os, LHCInfoPerFill beamInfo) {
  std::stringstream ss;
  beamInfo.print(ss);
  os << ss.str();
  return os;
}

bool LHCInfoPerFill::equals(const LHCInfoPerFill& rhs) const {
  if (m_isData != rhs.m_isData)
    return false;
  if (m_intParams != rhs.m_intParams)
    return false;
  if (m_floatParams != rhs.m_floatParams)
    return false;
  if (m_timeParams != rhs.m_timeParams)
    return false;
  if (m_stringParams != rhs.m_stringParams)
    return false;
  if (m_bunchConfiguration1 != rhs.m_bunchConfiguration1)
    return false;
  if (m_bunchConfiguration2 != rhs.m_bunchConfiguration2)
    return false;
  return true;
}

bool LHCInfoPerFill::empty() const { return m_intParams[0].empty(); }
