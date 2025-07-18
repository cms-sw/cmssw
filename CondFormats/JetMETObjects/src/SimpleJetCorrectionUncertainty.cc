#include "CondFormats/JetMETObjects/interface/SimpleJetCorrectionUncertainty.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>
#include <string>

/////////////////////////////////////////////////////////////////////////
SimpleJetCorrectionUncertainty::SimpleJetCorrectionUncertainty(const std::string& fDataFile) : mParameters{fDataFile} {}
/////////////////////////////////////////////////////////////////////////
SimpleJetCorrectionUncertainty::SimpleJetCorrectionUncertainty(const JetCorrectorParameters& fParameters)
    : mParameters{fParameters} {}
/////////////////////////////////////////////////////////////////////////
float SimpleJetCorrectionUncertainty::uncertainty(const std::vector<float>& fX, float fY, bool fDirection) const {
  float result = 1.;
  int bin = mParameters.binIndex(fX);
  if (bin < 0) {
    edm::LogError("SimpleJetCorrectionUncertainty") << " bin variables out of range";
    result = -999.0;
  } else
    result = uncertaintyBin((unsigned)bin, fY, fDirection);
  return result;
}
/////////////////////////////////////////////////////////////////////////
namespace {
  class Span3 {
  public:
    Span3(std::vector<float> const& iFrom, int offset = 0) : start_{iFrom.data() + offset}, size_{iFrom.size() / 3} {}
    float last() const { return fromIndex(size_ - 1); }
    float operator[](std::size_t i) const { return fromIndex(i); }
    std::size_t size() const { return size_; }

  private:
    float fromIndex(std::size_t i) const { return *(start_ + 3 * i); }
    float const* start_;
    std::size_t size_;
  };
  int findBin(const Span3& v, float x) {
    int n = v.size() - 1;
    if (n <= 0)
      return -1;
    if (x < v[0] || x >= v[n])
      return -1;
    for (int i = 0; i < n; i++) {
      if (x >= v[i] && x < v[i + 1])
        return i;
    }
    return 0;
  }
}  // namespace
float SimpleJetCorrectionUncertainty::uncertaintyBin(unsigned fBin, float fY, bool fDirection) const {
  if (fBin >= mParameters.size()) {
    edm::LogError("SimpleJetCorrectionUncertainty")
        << " wrong bin: " << fBin << ": only " << mParameters.size() << " are available";
    return -999.0;
  }
  const std::vector<float>& p = mParameters.record(fBin).parameters();
  if ((p.size() % 3) != 0 or p.empty())
    throw cms::Exception("SimpleJetCorrectionUncertainty")
        << "wrong # of parameters: multiple of 3 expected, " << p.size() << " got";
  Span3 yGrid{p};
  Span3 value{p, fDirection ? 1 : 2};
  float result = -1.0;
  if (fY <= yGrid[0])
    result = value[0];
  else if (fY >= yGrid.last())
    result = value.last();
  else {
    int bin = findBin(yGrid, fY);
    float vx[2], vy[2];
    for (int i = 0; i < 2; i++) {
      vx[i] = yGrid[bin + i];
      vy[i] = value[bin + i];
    }
    result = linearInterpolation(fY, vx, vy);
  }
  return result;
}
/////////////////////////////////////////////////////////////////////////
float SimpleJetCorrectionUncertainty::linearInterpolation(float fZ, const float fX[2], const float fY[2]) const {
  // Linear interpolation through the points (x[i],y[i]). First find the line that
  // is defined by the points and then calculate the y(z).
  float r = 0;
  if (fX[0] == fX[1]) {
    if (fY[0] == fY[1])
      r = fY[0];
    else {
      edm::LogError("SimpleJetCorrectionUncertainty") << " interpolation error";
      return -999.0;
    }
  } else {
    float a = (fY[1] - fY[0]) / (fX[1] - fX[0]);
    float b = (fY[0] * fX[1] - fY[1] * fX[0]) / (fX[1] - fX[0]);
    r = a * fZ + b;
  }
  return r;
}
