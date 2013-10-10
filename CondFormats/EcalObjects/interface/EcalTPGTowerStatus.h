#ifndef EcalTPGTowerStatus_h
#define EcalTPGTowerStatus_h

#include "CondFormats/Common/interface/Serializable.h"

#include <map>
#include <boost/cstdint.hpp>

class EcalTPGTowerStatus 
{
 public:
  EcalTPGTowerStatus() ;
  ~EcalTPGTowerStatus() ;

  // map<stripId, lut>
  const std::map<uint32_t, uint16_t> & getMap() const { return map_; }
  void  setValue(const uint32_t & id, const uint16_t & val) ;

 private:
  std::map<uint32_t, uint16_t> map_ ;


 COND_SERIALIZABLE;
};

typedef std::map<uint32_t, uint16_t>                 EcalTPGTowerStatusMap;
typedef std::map<uint32_t, uint16_t>::const_iterator EcalTPGTowerStatusMapIterator;

#endif
