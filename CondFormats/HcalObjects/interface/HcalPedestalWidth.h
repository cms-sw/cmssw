#ifndef HcalPedestalWidth_h
#define HcalPedestalWidth_h

/** 
\class HcalPedestalWidth
\author Fedor Ratnikov (UMd)
POOL object to store PedestalWidth values 4xCapId
$Author: ratnikov
$Date: 2005/12/15 23:38:04 $
$Revision: 1.1 $
*/

class HcalPedestalWidth {
 public:
  /// get value for all capId = 1..4, 10 values in total
  const float* getValues () const {return &mSigma11;}

  /// get width (sqrt(sigma_i_i)) for capId = 1..4
  float getWidth (int fCapId) const;

  /// get correlation element for capId1/2 = 1..4
  float getSigma (int fCapId1, int fCapId2) const;

  // functions below are not supposed to be used by consumer applications
  HcalPedestalWidth (int fId = 0);
  void setSigma (int fCapId1, int fCapId2, float fSigma);
  unsigned long rawId () const {return mId;}

 private:
  unsigned long mId;
  float mSigma11;
  float mSigma21;
  float mSigma22;
  float mSigma31;
  float mSigma32;
  float mSigma33;
  float mSigma41;
  float mSigma42;
  float mSigma43;
  float mSigma44;
};

#endif
