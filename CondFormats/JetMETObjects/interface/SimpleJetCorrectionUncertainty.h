#ifndef SimpleJetCorrectionUncertainty_h
#define SimpleJetCorrectionUncertainty_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <string>
#include <vector>
class JetCorrectorParameters;

class SimpleJetCorrectionUncertainty {
public:
  SimpleJetCorrectionUncertainty();
  SimpleJetCorrectionUncertainty(const std::string& fDataFile);
  SimpleJetCorrectionUncertainty(const JetCorrectorParameters& fParameters);
  SimpleJetCorrectionUncertainty(const SimpleJetCorrectionUncertainty&) = delete;
  SimpleJetCorrectionUncertainty& operator=(const SimpleJetCorrectionUncertainty&) = delete;
  ~SimpleJetCorrectionUncertainty();
  const JetCorrectorParameters& parameters() const { return *mParameters; }
  float uncertainty(const std::vector<float>& fX, float fY, bool fDirection) const;

private:
  int findBin(const std::vector<float>& v, float x) const;
  float uncertaintyBin(unsigned fBin, float fY, bool fDirection) const;
  float linearInterpolation(float fZ, const float fX[2], const float fY[2]) const;
  JetCorrectorParameters* mParameters;
};

#endif
