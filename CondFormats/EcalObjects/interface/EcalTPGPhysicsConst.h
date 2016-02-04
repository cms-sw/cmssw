#ifndef EcalTPGPhysicsConst_h
#define EcalTPGPhysicsConst_h

#include <map>
#include <boost/cstdint.hpp>

class EcalTPGPhysicsConst 
{
 public:
  EcalTPGPhysicsConst() ;
  ~EcalTPGPhysicsConst() ;

  struct Item 
  {
    double EtSat ; 
    double ttf_threshold_Low ; 
    double ttf_threshold_High ; 
    double FG_lowThreshold  ; 
    double FG_highThreshold ; 
    double FG_lowRatio ; 
    double FG_highRatio ; 
  } ;

  // first index is for barrel or endcap
  const std::map<uint32_t, Item> & getMap() const { return map_; }
  void  setValue(const uint32_t & id, const Item & value) ;

 private:
  std::map<uint32_t, Item> map_ ;

};

typedef std::map<uint32_t, EcalTPGPhysicsConst::Item>                 EcalTPGPhysicsConstMap;
typedef std::map<uint32_t, EcalTPGPhysicsConst::Item>::const_iterator EcalTPGPhysicsConstMapIterator;

#endif
