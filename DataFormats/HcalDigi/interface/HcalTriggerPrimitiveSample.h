#ifndef HCALTRIGGERPRIMITIVESAMPLE_H
#define HCALTRIGGERPRIMITIVESAMPLE_H 1

#include <boost/cstdint.hpp>
#include <ostream>

namespace cms {

/** \class HcalTriggerPrimitiveSample
    
   $Date: 2005/07/19 18:46:28 $
   $Revision: 1.1 $
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
  int fiber() const { return (theSample>>13)&0x7; }
  /// get the fiber channel number
  int fiberChan() const { return (theSample>>11)&0x3; }
  /// get the id channel
  int fiberAndChan() const { return (theSample>>11)&0x1F; }

private:
  uint16_t theSample;
};

}

std::ostream& operator<<(std::ostream& s, const cms::HcalTriggerPrimitiveSample& samp);

#endif
