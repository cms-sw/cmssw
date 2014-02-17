#ifndef HCALTRIGGERPRIMITIVESAMPLE_H
#define HCALTRIGGERPRIMITIVESAMPLE_H 1

#include <boost/cstdint.hpp>
#include <ostream>

/** \class HcalTriggerPrimitiveSample
    
  $Date: 2007/12/14 08:50:43 $
  $Revision: 1.7 $
  \author J. Mans - Minnesota
*/
class HcalTriggerPrimitiveSample {
public:
  HcalTriggerPrimitiveSample();
  HcalTriggerPrimitiveSample(uint16_t data);
  HcalTriggerPrimitiveSample(int encodedEt, bool finegrain, int slb, int slbchan);
  
  /// get the raw word
  uint16_t raw() const { return theSample; }
  /// get the encoded/compressed Et
  int compressedEt() const { return theSample&0xFF; }
  /// get the fine-grain bit
  bool fineGrain() const { return (theSample&0x100)!=0; }
  /// get the slb site number
  int slb() const { return ((theSample>>13)&0x7); }
  /// get the slb channel number
  int slbChan() const { return (theSample>>11)&0x3; }
  /// get the id channel
  int slbAndChan() const { return (theSample>>11)&0x1F; }
  
private:
  uint16_t theSample;
};

std::ostream& operator<<(std::ostream& s, const HcalTriggerPrimitiveSample& samp);


#endif
