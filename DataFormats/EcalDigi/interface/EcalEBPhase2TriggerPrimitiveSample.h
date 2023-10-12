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
  // The sample is a 18 bit word defined as:
  //
  //     o o o o o    o     o o o o o o o o o o o o
  //     |________|        |_______________________|
  //      ~60ps res  spike         Et
  //      time info  flag
  //

  /// get the raw word
  uint32_t raw() const { return theSample_ & 0x3ffff; }

  /// get the encoded Et (12 bits)
  int encodedEt() const { return (theSample_ & 0x3ffff) & 0xFFF; }

  bool l1aSpike() const { return (theSample_ & 0x3ffff & 0x1000) != 0; }

  int time() const { return (theSample_ & 0x3ffff) >> 13; }

  /// for streaming
  uint32_t operator()() { return theSample_ & 0x3ffff; }

private:
  uint32_t theSample_;
};

std::ostream& operator<<(std::ostream& s, const EcalEBPhase2TriggerPrimitiveSample& samp);

#endif
