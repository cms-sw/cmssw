#ifndef HCALTRIGGERPRIMITIVESAMPLE_H
#define HCALTRIGGERPRIMITIVESAMPLE_H 1

#include <boost/cstdint.hpp>
#include <ostream>

/** \class HcalTriggerPrimitiveSample
    
  $Date: 2005/10/04 13:37:35 $
  $Revision: 1.4 $
  \author J. Mans - Minnesota
*/
class HcalTriggerPrimitiveSample {
public:
  HcalTriggerPrimitiveSample();
  HcalTriggerPrimitiveSample(uint16_t data);
  HcalTriggerPrimitiveSample(int encodedEt, bool finegrain, int fiber, int fiberchan);
  
  /// get the raw word
  uint16_t raw() const { return theSample; }
  /// get the encoded/compressed Et
  int compressedEt() const { return theSample&0xFF; }
  /// get the fine-grain bit
  bool fineGrain() const { return (theSample&0x100)!=0; }
  /// get the fiber number
  int fiber() const { return ((theSample>>13)&0x7)+1; }
  /// get the fiber channel number
  int fiberChan() const { return (theSample>>11)&0x3; }
  /// get the id channel
  int fiberAndChan() const { return (theSample>>11)&0x1F; }
  
private:
  uint16_t theSample;
};

std::ostream& operator<<(std::ostream& s, const HcalTriggerPrimitiveSample& samp);


#endif
