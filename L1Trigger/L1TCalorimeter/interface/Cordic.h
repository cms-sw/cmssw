#ifndef Cordic_H
#define Cordic_H

#include <vector>
#include <stdint.h>

class Cordic
{
 public:
  Cordic( const uint32_t& aPhiScale , const uint32_t& aMagnitudeBits , const uint32_t& aSteps );
  virtual ~Cordic();

  void operator() ( int32_t aX , int32_t aY , int32_t& aPhi , uint32_t& aMagnitude );

  double NormalizePhi( const uint32_t& aPhi);
  double NormalizeMagnitude( const uint32_t& aMagnitude );
  int32_t IntegerizeMagnitude( const double& aMagnitude );

 private:
  uint32_t tower( const double& aRadians );

 private:
  uint32_t mPhiScale;
  uint32_t mMagnitudeScale;
  uint32_t mMagnitudeBits;
  uint32_t mSteps;
  uint32_t mMagnitudeRenormalization;
  std::vector<uint32_t> mRotations;

  const double mPi;
};

#endif
