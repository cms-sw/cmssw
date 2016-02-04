#ifndef EcalTPGSlidingWindow_h
#define EcalTPGSlidingWindow_h

#include <map>
#include <boost/cstdint.hpp>

class EcalTPGSlidingWindow 
{
 public:
  EcalTPGSlidingWindow() ;
  ~EcalTPGSlidingWindow() ;

  const std::map<uint32_t, uint32_t> & getMap() const { return map_; }
  void  setValue(const uint32_t & id, const uint32_t & value) ;

 private:
  std::map<uint32_t, uint32_t> map_ ;

};

typedef std::map<uint32_t, uint32_t>                 EcalTPGSlidingWindowMap;
typedef std::map<uint32_t, uint32_t>::const_iterator EcalTPGSlidingWindowMapIterator;

#endif
