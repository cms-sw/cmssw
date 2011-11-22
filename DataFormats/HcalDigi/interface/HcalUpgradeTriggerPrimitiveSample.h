#ifndef DIGIHCAL_HCALUPGRADETRIGGERPRIMITIVESAMPLE_H
#define DIGIHCAL_HCALUPGRADETRIGGERPRIMITIVESAMPLE_H

#include <boost/cstdint.hpp>
#include <ostream>

class HcalUpgradeTriggerPrimitiveSample {
  
 public:  
  
  HcalUpgradeTriggerPrimitiveSample();
  HcalUpgradeTriggerPrimitiveSample(uint32_t data);
  HcalUpgradeTriggerPrimitiveSample(int encodedEt, int fineGrain, int slb, int slbchan);
  
  uint32_t raw() const {return theSample; }  
  
  int slb            () const { return ((theSample >> 21) & 0x7 ); } // slb          has 3 bits, (22-24)
  int slbChan        () const { return ((theSample >> 19) & 0x3 ); } // slbchan      has 2 bits, (20-21)
  int slbAndChan     () const { return ((theSample >> 16) & 0x1F); } // slb AND chan has 5 bits, (17-21)
  int fineGrain      () const { return ((theSample >> 8 ) & 0xFF); } // fine grain   has 8 bits, (9-16)
  int compressedEt   () const { return ((theSample >> 0 ) & 0xFF); } // et           has 8 bits, (1 -8 )
  
 private:
  uint32_t theSample;

};

std::ostream& operator<<(std::ostream& s, const HcalUpgradeTriggerPrimitiveSample& samp);

#endif
