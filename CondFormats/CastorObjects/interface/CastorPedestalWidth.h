#ifndef CastorPedestalWidth_h
#define CastorPedestalWidth_h

/** 
\class CastorPedestalWidth
\author Fedor Ratnikov (UMd)
POOL object to store PedestalWidth values 4xCapId
$Author: ratnikov
$Date: 2009/03/26 18:03:15 $
$Revision: 1.2 $
Adapted for CASTOR by L. Mundim
*/
#include <boost/cstdint.hpp>

class CastorPedestalWidth {
 public:
  /// get value for all capId = 0..3, 10 values in total
  const float* getValues () const {return &mSigma00;}

  /// get width (sqrt(sigma_i_i)) for capId = 0..3
  float getWidth (int fCapId) const;

  /// get correlation element for capId1/2 = 0..3
  float getSigma (int fCapId1, int fCapId2) const;

  // functions below are not supposed to be used by consumer applications
  CastorPedestalWidth (int fId = 0);
  void setSigma (int fCapId1, int fCapId2, float fSigma);

  uint32_t rawId () const {return mId;}

  // produces pedestal noise in assumption of near correlations and small variations
  void makeNoise (unsigned fFrames, const double* fGauss, double* fNoise) const;

 private:
  uint32_t mId;
  float mSigma00;
  float mSigma01;
  float mSigma02;
  float mSigma03;
  float mSigma10;
  float mSigma11;
  float mSigma12;
  float mSigma13;
  float mSigma20;
  float mSigma21;
  float mSigma22;
  float mSigma23;
  float mSigma30;
  float mSigma31;
  float mSigma32;
  float mSigma33;
};

#endif
