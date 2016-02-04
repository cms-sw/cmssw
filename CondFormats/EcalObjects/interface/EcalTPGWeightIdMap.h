#ifndef EcalTPGWeightIdMap_h
#define EcalTPGWeightIdMap_h

#include <map>
#include <boost/cstdint.hpp>
#include "CondFormats/EcalObjects/interface/EcalTPGWeights.h"

class EcalTPGWeightIdMap
{
 public:
  typedef std::map<uint32_t, EcalTPGWeights> EcalTPGWeightMap ;
  typedef std::map<uint32_t, EcalTPGWeights>::const_iterator EcalTPGWeightMapItr ;

  EcalTPGWeightIdMap() ;
  ~EcalTPGWeightIdMap() ;

  const EcalTPGWeightMap & getMap() const { return map_; }
  void  setValue(const uint32_t & id, const  EcalTPGWeights & value) ;

 private:
  EcalTPGWeightMap map_ ;

};

#endif
