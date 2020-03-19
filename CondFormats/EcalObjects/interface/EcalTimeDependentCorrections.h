#ifndef CondFormats_EcalObjects_EcalTimeDependentCorrections_H
#define CondFormats_EcalObjects_EcalTimeDependentCorrections_H
/**
 *Author: Vladlen Timciuc, Caltech
 * Created: 10 July 2007
 * $Id: EcalLaserAPDPNRatios.h,v 1.6 2009/06/24 09:42:27 fra Exp $
 **/
#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include <vector>

class EcalTimeDependentCorrections {
public:
  struct Values {
    Values() : p1(0), p2(0), p3(0) {}
    float p1;
    float p2;
    float p3;

    COND_SERIALIZABLE;
  };
  struct Times {
    Times() : t1(0), t2(0), t3(0) {}
    edm::Timestamp t1;
    edm::Timestamp t2;
    edm::Timestamp t3;

    COND_SERIALIZABLE;
  };

  typedef EcalCondObjectContainer<Values> EcalValueMap;
  typedef std::vector<Times> EcalTimeMap;

  EcalTimeDependentCorrections() : time_map(92){};  // FIXME
  ~EcalTimeDependentCorrections(){};

  void setValue(uint32_t rawId, const Values& value) { value_map[rawId] = value; };
  const EcalValueMap& getValueMap() const { return value_map; }

  void setTime(int hashedIndex, const Times& value) { time_map[hashedIndex] = value; };
  const EcalTimeMap& getTimeMap() const { return time_map; }

private:
  EcalValueMap value_map;
  EcalTimeMap time_map;

  COND_SERIALIZABLE;
};

#endif
