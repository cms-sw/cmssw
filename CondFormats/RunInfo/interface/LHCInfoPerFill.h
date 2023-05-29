#ifndef CondFormats_RunInfo_LHCInfoPerFill_H
#define CondFormats_RunInfo_LHCInfoPerFill_H

#include "CondFormats/Serialization/interface/Serializable.h"
#include "CondFormats/RunInfo/interface/LHCInfoVectorizedFields.h"
#include "CondFormats/Common/interface/Time.h"
#include <bitset>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

class LHCInfoPerFill : public LHCInfoVectorizedFields {
public:
  enum FillType { UNKNOWN = 0, PROTONS, IONS, COSMICS, GAP };
  enum ParticleType { NONE = 0, PROTON, PB82, AR18, D, XE54 };

  enum IntParamIndex {
    LHC_FILL = 0,
    BUNCHES_1,
    BUNCHES_2,
    COLLIDING_BUNCHES,
    TARGET_BUNCHES,
    FILL_TYPE,
    PARTICLES_1,
    PARTICLES_2,
    ISIZE
  };
  enum FloatParamIndex {
    INTENSITY_1 = 0,
    INTENSITY_2,
    ENERGY,
    DELIV_LUMI,
    REC_LUMI,
    LUMI_PER_B,
    BEAM1_VC,
    BEAM2_VC,
    BEAM1_RF,
    BEAM2_RF,
    INST_LUMI,
    INST_LUMI_ERR,
    FSIZE
  };
  enum TimeParamIndex { CREATE_TIME = 0, BEGIN_TIME, END_TIME, TSIZE };
  enum StringParamIndex { INJECTION_SCHEME = 0, LHC_STATE, LHC_COMMENT, CTPPS_STATUS, SSIZE };

  typedef FillType FillTypeId;
  typedef ParticleType ParticleTypeId;
  LHCInfoPerFill();

  LHCInfoPerFill* cloneFill() const;

  //constant static unsigned integer hosting the maximum number of LHC bunch slots
  static size_t const bunchSlots = 3564;

  //constant static unsigned integer hosting the available number of LHC bunch slots
  static size_t const availableBunchSlots = 2808;

  void setFillNumber(unsigned short lhcFill);

  //getters
  unsigned short const fillNumber() const;

  unsigned short const bunchesInBeam1() const;

  unsigned short const bunchesInBeam2() const;

  unsigned short const collidingBunches() const;

  unsigned short const targetBunches() const;

  FillTypeId const fillType() const;

  ParticleTypeId const particleTypeForBeam1() const;

  ParticleTypeId const particleTypeForBeam2() const;

  float const intensityForBeam1() const;

  float const intensityForBeam2() const;

  float const energy() const;

  float const delivLumi() const;

  float const recLumi() const;

  float const instLumi() const;

  float const instLumiError() const;

  cond::Time_t const createTime() const;

  cond::Time_t const beginTime() const;

  cond::Time_t const endTime() const;

  std::string const& injectionScheme() const;

  std::vector<float> const& lumiPerBX() const;

  std::string const& lhcState() const;

  std::string const& lhcComment() const;

  std::string const& ctppsStatus() const;

  std::vector<float> const& beam1VC() const;

  std::vector<float> const& beam2VC() const;

  std::vector<float> const& beam1RF() const;

  std::vector<float> const& beam2RF() const;

  std::vector<float>& beam1VC();

  std::vector<float>& beam2VC();

  std::vector<float>& beam1RF();

  std::vector<float>& beam2RF();

  //returns a boolean, true if the injection scheme has a leading 25ns
  //TODO: parse the circulating bunch configuration, instead of the string.
  bool is25nsBunchSpacing() const;

  //returns a boolean, true if the bunch slot number is in the circulating bunch configuration
  bool isBunchInBeam1(size_t const& bunch) const;

  bool isBunchInBeam2(size_t const& bunch) const;

  //member functions returning *by value* a vector with all filled bunch slots
  std::vector<unsigned short> bunchConfigurationForBeam1() const;

  std::vector<unsigned short> bunchConfigurationForBeam2() const;

  //setters
  void setBunchesInBeam1(unsigned short const& bunches);

  void setBunchesInBeam2(unsigned short const& bunches);

  void setCollidingBunches(unsigned short const& collidingBunches);

  void setTargetBunches(unsigned short const& targetBunches);

  void setFillType(FillTypeId const& fillType);

  void setParticleTypeForBeam1(ParticleTypeId const& particleType);

  void setParticleTypeForBeam2(ParticleTypeId const& particleType);

  void setIntensityForBeam1(float const& intensity);

  void setIntensityForBeam2(float const& intensity);

  void setEnergy(float const& energy);

  void setDelivLumi(float const& delivLumi);

  void setRecLumi(float const& recLumi);

  void setInstLumi(float const& instLumi);

  void setInstLumiError(float const& instLumiError);

  void setCreationTime(cond::Time_t const& createTime);

  void setBeginTime(cond::Time_t const& beginTime);

  void setEndTime(cond::Time_t const& endTime);

  void setInjectionScheme(std::string const& injectionScheme);

  void setLumiPerBX(std::vector<float> const& lumiPerBX);

  void setLhcState(std::string const& lhcState);

  void setLhcComment(std::string const& lhcComment);

  void setCtppsStatus(std::string const& ctppsStatus);

  void setBeam1VC(std::vector<float> const& beam1VC);

  void setBeam2VC(std::vector<float> const& beam2VC);

  void setBeam1RF(std::vector<float> const& beam1RF);

  void setBeam2RF(std::vector<float> const& beam2RF);

  //sets all values in one go
  void setInfo(unsigned short const& bunches1,
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
               std::bitset<bunchSlots + 1> const& bunchConf2);

  bool equals(const LHCInfoPerFill& rhs) const;

  bool empty() const;

  //dumping values on output stream
  void print(std::stringstream& ss) const;

  std::bitset<bunchSlots + 1> const& bunchBitsetForBeam1() const;

  std::bitset<bunchSlots + 1> const& bunchBitsetForBeam2() const;

  void setBunchBitsetForBeam1(std::bitset<bunchSlots + 1> const& bunchConfiguration);

  void setBunchBitsetForBeam2(std::bitset<bunchSlots + 1> const& bunchConfiguration);

private:
  std::bitset<bunchSlots + 1> m_bunchConfiguration1, m_bunchConfiguration2;

  COND_SERIALIZABLE;
};

std::ostream& operator<<(std::ostream&, LHCInfoPerFill lhcInfoPerFill);

#endif  // CondFormats_RunInfo_LHCInfoPerFill_H
