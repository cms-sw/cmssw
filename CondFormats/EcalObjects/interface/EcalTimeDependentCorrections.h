#ifndef CondFormats_EcalObjects_EcalTimeDependentCorrections_H 
#define CondFormats_EcalObjects_EcalTimeDependentCorrections_H 
/**
 *Author: Vladlen Timciuc, Caltech
 * Created: 10 July 2007
 * $Id: EcalTimeDependentCorrections.h,v 1.1 2012/12/06 08:34:40 ferriff Exp $
 **/
#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include <vector>

class EcalTimeDependentCorrections {
 public:
  struct Values {
    float p1;
    float p2;
    float p3;
  };
  struct Times {
    edm::Timestamp t1;
    edm::Timestamp t2;
    edm::Timestamp t3;
  };

  typedef EcalCondObjectContainer<Values> EcalValueMap;
  typedef std::vector<Times> EcalTimeMap;

  EcalTimeDependentCorrections() : time_map(92) {}; // FIXME
  ~EcalTimeDependentCorrections() {};

  void  setValue(uint32_t rawId, const Values & value) { value_map[rawId] = value; };
  const EcalValueMap & getValueMap() const { return value_map; }
  
  void setTime(int hashedIndex, const Times & value) { time_map[hashedIndex] = value; };
  const EcalTimeMap & getTimeMap() const { return time_map; }  

 private:
  EcalValueMap value_map;
  EcalTimeMap time_map;
   
};

#endif
