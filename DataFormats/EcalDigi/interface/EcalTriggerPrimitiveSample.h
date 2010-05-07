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
  EcalTriggerPrimitiveSample(uint16_t data);
  EcalTriggerPrimitiveSample(int encodedEt, bool finegrain, int triggerFlag);

  ///Set data
  void setValue(uint16_t data){ theSample = data;}
  /// get the raw word
  uint16_t raw() const { return theSample; }
  /// get the encoded/compressed Et (8 bits)
  int compressedEt() const { return theSample&0xFF; }
  /// get the fine-grain bit (1 bit) 
  bool fineGrain() const { return (theSample&0x100)!=0; }
  /// get the Trigger tower Flag (3 bits)
  int ttFlag() const { return (theSample>>9)&0x7; }

  /// gets the L1A spike detection flag. 
  /// @return 1 if the trigger primitive was forced to zero because a spike was detected by L1 trigger,
  ///         0 otherwise
  int l1aSpike() const { return (theSample >>12) & 0x1; }
  
  /// for streaming
  uint16_t operator()() { return theSample; }

 private:
  uint16_t theSample;
};

std::ostream& operator<<(std::ostream& s, const EcalTriggerPrimitiveSample& samp);
  



#endif
