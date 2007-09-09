#ifndef CondFormats_EcalObjects_EcalLaserAPDPNRatios_H 
#define CondFormats_EcalObjects_EcalLaserAPDPNRatios_H 
/**
 *Author: Vladlen Timciuc, Caltech
 * Created: 10 July 2007
 * $Id: EcalLaserAPDPNRatios.h,v 1.3 2007/07/27 13:56:53 xiezhen Exp $
 **/
#include <vector>
#include <boost/cstdint.hpp>
#include "DataFormats/Provenance/interface/Timestamp.h"

class EcalLaserAPDPNRatios {
 public:
  struct EcalLaserAPDPNpair{
    float p1;
    float p2;
  };
  struct EcalLaserTimeStamp{
    edm::Timestamp t1;
    edm::Timestamp t2;
  };
  
  typedef std::vector<EcalLaserAPDPNpair> EcalLaserAPDPNRatiosMap;
  typedef std::vector<EcalLaserTimeStamp> EcalLaserTimeStampMap;

  EcalLaserAPDPNRatios();
  ~EcalLaserAPDPNRatios();
   
  void  setValue(int hashedIndex, const EcalLaserAPDPNpair& value) { laser_map[hashedIndex] = value; };
  const EcalLaserAPDPNRatiosMap& getLaserMap() const { return laser_map; }
  
  void setTime(int hashedIndex, const EcalLaserTimeStamp& value) { time_map[hashedIndex] = value; };
  const EcalLaserTimeStampMap& getTimeMap() const { return time_map; }  

 private:
  EcalLaserAPDPNRatiosMap laser_map;
  EcalLaserTimeStampMap time_map;
   
};

#endif
