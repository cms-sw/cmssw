#ifndef CondFormats_EcalObjects_EcalLaserAPDPNRatios_H 
#define CondFormats_EcalObjects_EcalLaserAPDPNRatios_H 
/**
 *Author: Vladlen Timciuc, Caltech
 * Created: 10 July 2007
 * $Id: EcalLaserAPDPNRatios.cc, 1.2 2007/07/10 14:12:00 Vladlen Exp $
 **/
#include <map>
#include <boost/cstdint.hpp>


class EcalLaserAPDPNRatios {
 public:
  struct EcalLaserAPDPNpair{
    float p1;
    float p2;
  };
  struct EcalLaserTimeStamp{
    uint32_t t1;
    uint32_t t2;
  };
  
  typedef std::map<uint32_t, EcalLaserAPDPNpair> EcalLaserAPDPNRatiosMap;
  typedef std::map<uint32_t, EcalLaserTimeStamp> EcalLaserTimeStampMap;
  typedef std::map<uint32_t, EcalLaserAPDPNpair>::const_iterator EcalLaserAPDPNRatiosMapIterator;
  typedef std::map<uint32_t, EcalLaserTimeStamp>::const_iterator EcalLaserTimeStampMapIterator;

  EcalLaserAPDPNRatios();
  ~EcalLaserAPDPNRatios();

   
  void  setValue(const uint32_t& id, const EcalLaserAPDPNpair& value);
  const EcalLaserAPDPNRatiosMap& getLaserMap() const { return laser_map; }
  
  void setTime(const int& id, const EcalLaserTimeStamp& value);
  const EcalLaserTimeStampMap& getTimeMap() const { return time_map; }
  

 private:
  EcalLaserAPDPNRatiosMap laser_map;
  EcalLaserTimeStampMap time_map;
   
};

#endif
