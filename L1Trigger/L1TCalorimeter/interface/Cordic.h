#ifndef Cordic_H
#define Cordic_H

#include <vector>
#include <cstdint>

class Cordic {
public:
  Cordic(const uint32_t& aPhiScale, const uint32_t& aMagnitudeBits, const uint32_t& aSteps);
  virtual ~Cordic();

  void operator()(int32_t aX, int32_t aY, int32_t& aPhi, uint32_t& aMagnitude) const;

  double NormalizePhi(const uint32_t& aPhi) const;
  double NormalizeMagnitude(const uint32_t& aMagnitude) const;
  int32_t IntegerizeMagnitude(const double& aMagnitude) const;

private:
  uint32_t tower(const double& aRadians) const;

private:
  uint32_t mPhiScale;
  uint32_t mMagnitudeScale;
  uint32_t mMagnitudeBits;
  uint32_t mSteps;
  uint64_t mMagnitudeRenormalization;
  std::vector<uint32_t> mRotations;

  const double mPi;
};

#endif
