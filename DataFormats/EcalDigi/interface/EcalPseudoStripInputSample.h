#ifndef ECALPSEUDOSTRIPINPUTSAMPLE_H
#define ECALPSEUDOSTRIPINPUTSAMPLE_H

#include <ostream>
#include <cstdint>

/** \class EcalPseudoStripInputSample
      

*/

class EcalPseudoStripInputSample {
public:
  EcalPseudoStripInputSample();
  EcalPseudoStripInputSample(uint16_t data);
  EcalPseudoStripInputSample(int pseudoStripInput, bool finegrain);

  ///Set data
  void setValue(uint16_t data) { theSample = data; }
  /// get the raw word
  uint16_t raw() const { return theSample; }
  /// get the pseudoStrip Input amplitude (12 bits)
  int pseudoStripInput() const { return theSample & 0xFFF; }
  /// get the fine-grain bit (1 bit, the 13-th)
  bool fineGrain() const { return (theSample & 0x1000) != 0; }

  /// for streaming
  uint16_t operator()() { return theSample; }

private:
  uint16_t theSample;
};

std::ostream& operator<<(std::ostream& s, const EcalPseudoStripInputSample& samp);

#endif
