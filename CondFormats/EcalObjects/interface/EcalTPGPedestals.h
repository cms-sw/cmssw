#ifndef EcalTPGPedestals_h
#define EcalTPGPedestals_h

#include <map>
#include <boost/cstdint.hpp>

class EcalTPGPedestals 
{
 public:
  EcalTPGPedestals() ;
  ~EcalTPGPedestals() ;

  struct Item 
  {
    uint32_t mean_x12 ;
    uint32_t mean_x6 ;
    uint32_t mean_x1 ;
  };

  const std::map<uint32_t, Item> & getMap() const { return map_; }
  void  setValue(const uint32_t & id, const Item & value) ;

 private:
  std::map<uint32_t, Item> map_ ;

};

typedef std::map<uint32_t, EcalTPGPedestals::Item>                 EcalTPGPedestalsMap;
typedef std::map<uint32_t, EcalTPGPedestals::Item>::const_iterator EcalTPGPedestalsMapIterator;

#endif
