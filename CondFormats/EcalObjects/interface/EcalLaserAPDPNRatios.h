#ifndef CondFormats_EcalObjects_EcalLaserAPDPNRatios_H 
#define CondFormats_EcalObjects_EcalLaserAPDPNRatios_H 
/**
 *Author: Vladlen Timciuc, Caltech
 * Created: 10 July 2007
 * $Id: EcalLaserAPDPNRatios.h,v 1.6 2009/06/24 09:42:27 fra Exp $
 **/
#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include <vector>

class EcalLaserAPDPNRatios {
 public:
  struct EcalLaserAPDPNpair{
    float p1;
    float p2;
    float p3;
  };
  struct EcalLaserTimeStamp{
    edm::Timestamp t1;
    edm::Timestamp t2;
    edm::Timestamp t3;
  };
  
  typedef EcalCondObjectContainer<EcalLaserAPDPNpair> EcalLaserAPDPNRatiosMap;
  typedef std::vector<EcalLaserTimeStamp> EcalLaserTimeStampMap;

  EcalLaserAPDPNRatios() : time_map(92) {}; // FIXME
  ~EcalLaserAPDPNRatios() {};
   
  void  setValue(uint32_t rawId, const EcalLaserAPDPNpair& value) { laser_map[rawId] = value; };
  const EcalLaserAPDPNRatiosMap& getLaserMap() const { return laser_map; }
  
  void setTime(int hashedIndex, const EcalLaserTimeStamp& value) { time_map[hashedIndex] = value; };
  const EcalLaserTimeStampMap& getTimeMap() const { return time_map; }  

 private:
  EcalLaserAPDPNRatiosMap laser_map;
  EcalLaserTimeStampMap time_map;
   
};

#endif
