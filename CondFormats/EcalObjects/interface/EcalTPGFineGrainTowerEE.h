#ifndef EcalTPGFineGrainTowerEE_h
#define EcalTPGFineGrainTowerEE_h

#include <map>
#include <boost/cstdint.hpp>

class EcalTPGFineGrainTowerEE 
{
 public:
  EcalTPGFineGrainTowerEE() ;
  ~EcalTPGFineGrainTowerEE() ;

  // map<stripId, lut>
  const std::map<uint32_t, uint32_t> & getMap() const { return map_; }
  void  setValue(const uint32_t & id, const uint32_t & lut) ;

 private:
  std::map<uint32_t, uint32_t> map_ ;

};

typedef std::map<uint32_t, uint32_t>                 EcalTPGFineGrainTowerEEMap;
typedef std::map<uint32_t, uint32_t>::const_iterator EcalTPGFineGrainTowerEEMapIterator;

#endif
