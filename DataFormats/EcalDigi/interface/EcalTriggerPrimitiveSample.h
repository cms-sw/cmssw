#ifndef ECALTRIGGERPRIMITIVESAMPLE_H
#define ECALTRIGGERPRIMITIVESAMPLE_H 1

#include <ostream>
#include <cstdint>

/** \class EcalTriggerPrimitiveSample
      

*/

class EcalTriggerPrimitiveSample {
public:
  EcalTriggerPrimitiveSample();
  EcalTriggerPrimitiveSample(uint16_t data);
  EcalTriggerPrimitiveSample(int encodedEt, bool finegrain, int triggerFlag);
  EcalTriggerPrimitiveSample(int encodedEt, bool finegrain, int stripFGVB, int triggerFlag);

  ///Set data
  void setValue(uint16_t data) { theSample = data; }
  /// get the raw word
  uint16_t raw() const { return theSample; }
  /// get the encoded/compressed Et (8 bits)
  int compressedEt() const { return theSample & 0xFF; }
  /// get the fine-grain bit (1 bit)
  bool fineGrain() const { return (theSample & 0x100) != 0; }
  /// get the Trigger tower Flag (3 bits)
  int ttFlag() const { return (theSample >> 9) & 0x7; }

  /// Gets the L1A spike detection flag. Beware the flag is inverted.
  /// Deprecated, use instead sFGVB() method, whose name is less missleading
  /// @return 0 spike like pattern
  ///         1 EM shower like pattern
  int l1aSpike() const { return (theSample >> 12) & 0x1; }

  /// Gets the "strip fine grain veto bit" (sFGVB) used as L1A spike detection
  /// @return 0 spike like pattern
  ///         1 EM shower like pattern
  int sFGVB() const { return (theSample >> 12) & 0x1; }

  /// for streaming
  uint16_t operator()() { return theSample; }

private:
  uint16_t theSample;
};

std::ostream& operator<<(std::ostream& s, const EcalTriggerPrimitiveSample& samp);

#endif
