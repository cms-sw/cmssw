#ifndef CondFormats_EcalObjects_EcalLaserAPDPNRatios_H
#define CondFormats_EcalObjects_EcalLaserAPDPNRatios_H
/**
 *Author: Vladlen Timciuc, Caltech
 * Created: 10 July 2007
 * $Id: EcalLaserAPDPNRatios.h,v 1.5 2007/09/27 09:42:55 ferriff Exp $
 **/
#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include <vector>

class EcalLaserAPDPNRatios {
public:
  struct EcalLaserAPDPNpair {
    EcalLaserAPDPNpair() : p1(0), p2(0), p3(0) {}
    float p1;
    float p2;
    float p3;

    COND_SERIALIZABLE;
  };
  struct EcalLaserTimeStamp {
    EcalLaserTimeStamp() : t1(), t2(), t3() {}
    edm::Timestamp t1;
    edm::Timestamp t2;
    edm::Timestamp t3;

    COND_SERIALIZABLE;
  };

  typedef EcalCondObjectContainer<EcalLaserAPDPNpair> EcalLaserAPDPNRatiosMap;
  typedef std::vector<EcalLaserTimeStamp> EcalLaserTimeStampMap;

  EcalLaserAPDPNRatios() : time_map(92){};  // FIXME
  ~EcalLaserAPDPNRatios(){};

  void setValue(uint32_t rawId, const EcalLaserAPDPNpair& value) { laser_map[rawId] = value; };
  const EcalLaserAPDPNRatiosMap& getLaserMap() const { return laser_map; }

  void setTime(int hashedIndex, const EcalLaserTimeStamp& value) { time_map[hashedIndex] = value; };
  const EcalLaserTimeStampMap& getTimeMap() const { return time_map; }

private:
  EcalLaserAPDPNRatiosMap laser_map;
  EcalLaserTimeStampMap time_map;

  COND_SERIALIZABLE;
};

#endif
