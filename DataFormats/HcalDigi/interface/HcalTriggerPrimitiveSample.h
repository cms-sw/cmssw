#ifndef HCALTRIGGERPRIMITIVESAMPLE_H
#define HCALTRIGGERPRIMITIVESAMPLE_H 1

#include <ostream>
#include <cstdint>

/** \class HcalTriggerPrimitiveSample
    
  \author J. Mans - Minnesota
*/
class HcalTriggerPrimitiveSample {
public:
  HcalTriggerPrimitiveSample();
  HcalTriggerPrimitiveSample(uint16_t data);
  HcalTriggerPrimitiveSample(int encodedEt, bool finegrain, int slb, int slbchan);
  HcalTriggerPrimitiveSample(int encodedEt, int finegrainExtended);

  /// get the raw word
  uint16_t raw() const { return theSample; }
  /// get the encoded/compressed Et
  int compressedEt() const { return theSample & 0xFF; }
  /// get fine-grain bit (traditional)
  bool fineGrain(int i = 0) const { return (((theSample) >> (i + 8)) & 0x1) != 0; }

private:
  uint16_t theSample;
};

std::ostream& operator<<(std::ostream& s, const HcalTriggerPrimitiveSample& samp);

#endif
