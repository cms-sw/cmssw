#ifndef DATAFORMATS_SISTRIPAPPROXIMATECLUSTERv1_H
#define DATAFORMATS_SISTRIPAPPROXIMATECLUSTERv1_H

#include <vector>
#include <numeric>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class SiStripApproximateClusterv1  {
public:
  
  SiStripApproximateClusterv1() {}

  explicit SiStripApproximateClusterv1(uint8_t avgCharge, uint16_t barycenter, uint8_t width):avgCharge_(avgCharge) {
    baryWidth_ = barycenter;
    baryWidth_ = baryWidth_ << 6;
    if(width>0x3F) width=0x3F;
    baryWidth_ += width;
  }

  uint16_t barycenter() const {return (uint16_t)((baryWidth_ & 0xFFC0) >> 6);}
  uint8_t width() const {return (uint8_t) (baryWidth_ & 0x3F);}
  uint16_t baryWidth() const {return baryWidth_;}
  uint8_t  avgCharge() const{return avgCharge_;} 

private:

  uint16_t                baryWidth_ = 0;

  uint8_t                 avgCharge_ = 0;
};
#endif // DATAFORMATS_SISTRIPAPPROXIMATECLUSTERv1_H
