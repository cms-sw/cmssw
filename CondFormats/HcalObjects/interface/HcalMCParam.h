#ifndef HcalMCParam_h
#define HcalMCParam_h

/** 
\class HcalMCParam
\author Radek Ofierzynski
POOL object to store MC information
*/

#include <boost/cstdint.hpp>

// definition 8.Feb.2011
// MC signal shape integer variable assigned to each readout this way:
// 0 - regular HPD  HB/HE/HO shape
// 1 - "special" HB shape
// 2 - SiPMs shape (HO, possibly also in HB/HE)
// 3 - HF Shape
// 4 - ZDC shape 


class HcalMCParam {
 public:
  HcalMCParam():mId(0), mSignalShape(0) {}

  HcalMCParam(unsigned long fId, unsigned int fSignalShape):
    mId(fId), mSignalShape(fSignalShape) {}

  uint32_t rawId () const {return mId;}

  unsigned int signalShape() const {return mSignalShape;}

 private:
  uint32_t mId;
  uint32_t mSignalShape;
};

#endif
