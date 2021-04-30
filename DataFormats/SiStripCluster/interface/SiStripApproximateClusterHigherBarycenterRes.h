#ifndef DATAFORMATS_SiStripApproximateClusterHigherBarycenterRes_H
#define DATAFORMATS_SiStripApproximateClusterHigherBarycenterRes_H

#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

class SiStripApproximateClusterHigherBarycenterRes  {
public:
  
  SiStripApproximateClusterHigherBarycenterRes() {}
  explicit SiStripApproximateClusterHigherBarycenterRes(const SiStripCluster& cluster){
    rawBarycenter_=static_cast<uint16_t>(std::round(cluster.barycenter()*8));
    width_=cluster.size();
    if(width_>0x3F) width_=0x3F;
    avgCharge_ = static_cast<uint8_t>(cluster.charge()/cluster.size());

    std::cout << " rawBaricenter " << rawBarycenter_ <<
      " barycenter: " << this->barycenter() << 
      " OrigBarycenter: " << cluster.barycenter() << 
      " OrigBarycenterCasted: " << static_cast<uint16_t>(cluster.barycenter()) << 
      " OrigBarycenterCastedRounded: " << static_cast<uint16_t>(cluster.barycenter())<< 
      " ratio " << barycenter()/cluster.barycenter() << std::endl;
  }
  

  float barycenter() const {return (float)rawBarycenter_/8;}
  uint16_t rawBarycenter() const {return rawBarycenter_;}
  uint8_t width() const {return width_;}
  uint8_t  avgCharge() const{return avgCharge_;} 

private:
  uint16_t                rawBarycenter_ = 0;
  uint8_t                 width_=0;
  uint8_t                 avgCharge_ = 0;
};
#endif // DATAFORMATS_SiStripApproximateClusterHigherBarycenterRes_H
