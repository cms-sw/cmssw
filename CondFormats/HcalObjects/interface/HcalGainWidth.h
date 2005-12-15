#ifndef HcalGainWidth_h
#define HcalGainWidth_h

/** 
\class HcalGainWidth
\author Fedor Ratnikov (UMd)
POOL object to store GainWidth values 4xCapId
$Author: ratnikov
$Date: 2005/10/06 21:25:32 $
$Revision: 1.5 $
*/

class HcalGainWidth {
 public:
  /// get value for all capId = 1..4
  const float* getValues () const {return &mValue1;}
  /// get value for capId = 1..4
  float getValue (int fCapId) const {return *(getValues () + (fCapId - 1));}

  // functions below are not supposed to be used by consumer applications

  HcalGainWidth () : mId (0), mValue1 (0), mValue2 (0), mValue3 (0), mValue4 (0) {}
  
  HcalGainWidth (unsigned long fId, float fCap1, float fCap2, float fCap3, float fCap4) :
    mId (fId),
    mValue1 (fCap1),
    mValue2 (fCap2),
    mValue3 (fCap3),
    mValue4 (fCap4) {}

  unsigned long rawId () const {return mId;}

 private:
  unsigned long mId;
  float mValue1;
  float mValue2;
  float mValue3;
  float mValue4;
};

#endif
