#ifndef DATAFORMATS_SISTRIPAPPROXIMATECLUSTER_H
#define DATAFORMATS_SISTRIPAPPROXIMATECLUSTER_H

#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

class SiStripApproximateCluster  {
public:
  
  SiStripApproximateCluster() {}
  explicit SiStripApproximateCluster(const SiStripCluster& cluster){
    rawBarycenter_=static_cast<uint16_t>(std::round(cluster.barycenter()));
    width_=cluster.size();
    if(width_>0x3F) width_=0x3F;
    avgCharge_ = static_cast<uint8_t>(cluster.charge()/cluster.size());
  }
  

  uint16_t barycenter() const {return rawBarycenter_;}
  uint8_t width() const {return width_;}
  uint8_t  avgCharge() const{return avgCharge_;} 

private:
  uint16_t                rawBarycenter_ = 0;
  uint8_t                 width_=0;
  uint8_t                 avgCharge_ = 0;
};
#endif // DATAFORMATS_SiStripApproximateCluster_H
