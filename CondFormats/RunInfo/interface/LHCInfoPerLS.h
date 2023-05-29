#ifndef CondFormats_RunInfo_LHCInfoPerLS_H
#define CondFormats_RunInfo_LHCInfoPerLS_H

#include "CondFormats/Serialization/interface/Serializable.h"
#include "CondFormats/RunInfo/interface/LHCInfoVectorizedFields.h"

#include "CondCore/CondDB/interface/Types.h"

#include <bitset>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

class LHCInfoPerLS : public LHCInfoVectorizedFields {
public:
  enum IntParamIndex { LHC_FILL = 0, LUMI_SECTION, ISIZE };
  enum FloatParamIndex { CROSSING_ANGLE_X = 0, CROSSING_ANGLE_Y, BETA_STAR_X, BETA_STAR_Y, FSIZE };
  enum TimeParamIndex { RUN_NUMBER = 0, TSIZE };

  LHCInfoPerLS();

  //getters
  unsigned short const fillNumber() const;

  unsigned int const& lumiSection() const;

  float const crossingAngleX() const;

  float const crossingAngleY() const;

  float const betaStarX() const;

  float const betaStarY() const;

  cond::Time_t runNumber() const;

  //setters
  void setFillNumber(unsigned short const& lhcFill);

  void setLumiSection(unsigned int const& lumiSection);

  void setCrossingAngleX(float const& angle);

  void setCrossingAngleY(float const& angle);

  void setBetaStarX(float const& betaStar);

  void setBetaStarY(float const& betaStar);

  void setRunNumber(cond::Time_t const& runNumber);

  //sets all values in one go
  void setInfo(unsigned short const& lhcFill,
               unsigned int const& lumiSection,
               float const& angleX,
               float const& angleY,
               float const& betaX,
               float const& betaY,
               cond::Time_t const& runNumber);

  bool equals(const LHCInfoPerLS& rhs) const;

  bool empty() const;

  //dumping values on output stream
  void print(std::stringstream& ss) const;

  COND_SERIALIZABLE;
};

std::ostream& operator<<(std::ostream&, LHCInfoPerLS lhcInfo);

#endif  // CondFormats_RunInfo_LHCInfoPerLS_H
