#ifndef SimpleJetCorrectionUncertainty_h
#define SimpleJetCorrectionUncertainty_h

#include <string>
#include <vector>
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"

class SimpleJetCorrectionUncertainty {
public:
  SimpleJetCorrectionUncertainty() = default;
  SimpleJetCorrectionUncertainty(const std::string& fDataFile);
  SimpleJetCorrectionUncertainty(const JetCorrectorParameters& fParameters);
  SimpleJetCorrectionUncertainty(const SimpleJetCorrectionUncertainty&) = delete;
  SimpleJetCorrectionUncertainty& operator=(const SimpleJetCorrectionUncertainty&) = delete;
  SimpleJetCorrectionUncertainty& operator=(SimpleJetCorrectionUncertainty&&) = default;
  ~SimpleJetCorrectionUncertainty() = default;
  const JetCorrectorParameters& parameters() const { return mParameters; }
  float uncertainty(const std::vector<float>& fX, float fY, bool fDirection) const;

private:
  float uncertaintyBin(unsigned fBin, float fY, bool fDirection) const;
  float linearInterpolation(float fZ, const float fX[2], const float fY[2]) const;
  JetCorrectorParameters mParameters;
};

#endif
