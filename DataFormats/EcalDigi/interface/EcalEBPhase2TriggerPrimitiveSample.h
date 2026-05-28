#ifndef DataFormats_EcalDig_EcalEBPhase2TriggerPrimitiveSample_h
#define DataFormats_EcalDig_EcalEBPhase2TriggerPrimitiveSample_h

#include <ostream>
#include <cstdint>

/** \class EcalEBPhase2TriggerPrimitiveSample
\author N. Marinelli - Univ of Notre Dame

*/

class EcalEBPhase2TriggerPrimitiveSample {
public:
  EcalEBPhase2TriggerPrimitiveSample();
  EcalEBPhase2TriggerPrimitiveSample(uint32_t data);
  EcalEBPhase2TriggerPrimitiveSample(int encodedEt);
  EcalEBPhase2TriggerPrimitiveSample(int encodedEt, bool isASpike);
  EcalEBPhase2TriggerPrimitiveSample(int encodedEt, bool isASpike, int timing);

  ///Set data
  void setValue(uint32_t data) { theSample_ = data; }
  // The sample is a 16 bit word defined as:
  //
  //     o o o o o    o     o o o o o o o o o o
  //     |________|        |___________________|
  //      ~60ps res  spike         Et
  //      time info  flag
  //

  /// get the raw word
  uint32_t raw() const { return theSample_ & 0xffff; }

  /// get the encoded Et (10 bits)
  int encodedEt() const { return (raw() ) & 0x3FF; }

  bool l1aSpike() const { return (raw() & 0x400) != 0; }

  int time() const { return raw() >> 11; }

  /// for streaming
  uint32_t operator()() { return raw(); }

private:
  uint32_t theSample_;
};

std::ostream& operator<<(std::ostream& s, const EcalEBPhase2TriggerPrimitiveSample& samp);

#endif
