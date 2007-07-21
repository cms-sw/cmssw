#ifndef ECALTRIGGERPRIMITIVESAMPLE_H
#define ECALTRIGGERPRIMITIVESAMPLE_H 1

#include <boost/cstdint.hpp>
#include <ostream>



/** \class EcalTriggerPrimitiveSample
      
$Id : $

*/

class EcalTriggerPrimitiveSample {
 public:
  EcalTriggerPrimitiveSample();
  EcalTriggerPrimitiveSample(const uint16_t& data);
  EcalTriggerPrimitiveSample(const int& encodedEt, const bool& finegrain, const int& triggerFlag);

  ///Set data
  void setValue(const uint16_t& data){ theSample = data;}
  /// get the raw word
  uint16_t raw() const { return theSample; }
  /// get the encoded/compressed Et (8 bits)
  int compressedEt() const { return theSample&0xFF; }
  /// get the fine-grain bit (1 bit) 
  bool fineGrain() const { return (theSample&0x100)!=0; }
  /// get the Trigger tower Flag (3 bits)
  int ttFlag() const { return (theSample>>9)&0x7; }
    
  /// for streaming
  uint16_t operator()() { return theSample; }

 private:
  uint16_t theSample;
};

std::ostream& operator<<(std::ostream& s, const EcalTriggerPrimitiveSample& samp);
  



#endif
