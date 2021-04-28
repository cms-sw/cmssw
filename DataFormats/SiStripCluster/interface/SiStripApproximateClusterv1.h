#ifndef DATAFORMATS_SISTRIPAPPROXIMATECLUSTERv1_H
#define DATAFORMATS_SISTRIPAPPROXIMATECLUSTERv1_H

#include <numeric>
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

class SiStripApproximateClusterv1  {
public:
  
  SiStripApproximateClusterv1() {}
  explicit SiStripApproximateClusterv1(const SiStripCluster& cluster){
    barycenter_=static_cast<uint16_t>(cluster.barycenter());
    width_=cluster.size();
    avgCharge_ = static_cast<uint8_t>(cluster.charge()/cluster.size());
  }
  
  explicit SiStripApproximateClusterv1(uint8_t avgCharge, uint16_t barycenter, uint8_t width):avgCharge_(avgCharge) {
    barycenter_ = barycenter;
    width_ = width;
    if(width_>0x3F) width_=0x3F;
  }

  uint16_t barycenter() const {return barycenter_;}
  uint8_t width() const {return width_;}
  uint8_t  avgCharge() const{return avgCharge_;} 

private:
  uint16_t                barycenter_ = 0;
  uint8_t                 width_=0;
  uint8_t                 avgCharge_ = 0;
};
#endif // DATAFORMATS_SISTRIPAPPROXIMATECLUSTERv1_H
