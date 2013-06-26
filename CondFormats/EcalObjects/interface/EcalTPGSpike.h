#ifndef EcalTPGSpike_h
#define EcalTPGSpike_h

#include <map>
#include <boost/cstdint.hpp>

class EcalTPGSpike 
{
 public:
  typedef std::map<uint32_t, uint16_t>                 EcalTPGSpikeMap;
  typedef std::map<uint32_t, uint16_t>::const_iterator EcalTPGSpikeMapIterator;

  EcalTPGSpike() ;
  ~EcalTPGSpike() ;

  // map<stripId, lut>
  const std::map<uint32_t, uint16_t> & getMap() const { return map_; }
  void  setValue(const uint32_t & id, const uint16_t & val) ;

 private:
  std::map<uint32_t, uint16_t> map_ ;

};

#endif
