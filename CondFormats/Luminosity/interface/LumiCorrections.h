#ifndef CondFormats_Luminosity_LumiCorrections_h
#define CondFormats_Luminosity_LumiCorrections_h

/** 
 * \class LumiCorrections
 * 
 * \author Sam Higginbotham, Chris Palmer
 *  
 *  This class should contain scale factors for correcting
 *  out-of-time pile-up per bunch crossing on rates intended
 *  for luminosity estimates.  There is the option of saving 
 *  the total scale factor on the total luminosity in
 *  m_overallCorrection as well.  
 */

#include <sstream>
#include <cstring>
#include <vector>
#include <boost/serialization/vector.hpp>
#include "CondFormats/Serialization/interface/Serializable.h"

class LumiCorrections {
public:
  void setOverallCorrection(float overallCorrection) { m_overallCorrection = overallCorrection; }
  void setType1Fraction(float type1frac) { m_type1Fraction = type1frac; }
  void setType1Residual(float type1res) { m_type1Residual = type1res; }
  void setType2Residual(float type2res) { m_type2Residual = type2res; }
  void setCorrectionsBX(std::vector<float>& correctBX) { m_correctionsBX.assign(correctBX.begin(), correctBX.end()); }
  float getOverallCorrection() { return m_overallCorrection; }
  float getCorrectionAtBX(float bx) { return m_correctionsBX[bx]; }
  float getType1Fraction() { return m_type1Fraction; }
  float getType1Residual() { return m_type1Residual; }
  float getType2Residual() { return m_type2Residual; }
  const std::vector<float>& getCorrectionsBX() const { return m_correctionsBX; }

private:
  float m_overallCorrection;
  float m_type1Fraction;
  float m_type1Residual;
  float m_type2Residual;
  std::vector<float> m_correctionsBX;
  COND_SERIALIZABLE;
};
#endif
