#ifndef ECALEBTRIGGERPRIMITIVESAMPLE_H
#define ECALEBTRIGGERPRIMITIVESAMPLE_H 1

#include <ostream>
#include <cstdint>

/** \class EcalEBTriggerPrimitiveSample
\author N. Marinelli - Univ of Notre Dame

*/

class EcalEBTriggerPrimitiveSample {
public:
  EcalEBTriggerPrimitiveSample();
  EcalEBTriggerPrimitiveSample(uint16_t data);
  EcalEBTriggerPrimitiveSample(int encodedEt);
  EcalEBTriggerPrimitiveSample(int encodedEt, bool isASpike);
  EcalEBTriggerPrimitiveSample(int encodedEt, bool isASpike, int timing);

  ///Set data
  void setValue(uint16_t data) { theSample = data; }
  // The sample is a 16 bit word defined as:
  //
  //     o o o o o    o     o o o o o o o o o o
  //     |________|        |____________________|
  //      ~60ps res  spike         Et
  //      time info  flag
  //

  /// get the raw word
  uint16_t raw() const { return theSample; }

  /// get the encoded Et (10 bits)
  int encodedEt() const { return theSample & 0x3FF; }

  bool l1aSpike() const { return (theSample & 0x400) != 0; }

  int time() const { return theSample >> 11; }

  /// for streaming
  uint16_t operator()() { return theSample; }

private:
  uint16_t theSample;
};

std::ostream& operator<<(std::ostream& s, const EcalEBTriggerPrimitiveSample& samp);

#endif
