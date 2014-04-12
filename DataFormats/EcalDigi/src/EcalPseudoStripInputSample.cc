#include "DataFormats/EcalDigi/interface/EcalPseudoStripInputSample.h"


EcalPseudoStripInputSample::EcalPseudoStripInputSample() : theSample(0) { }
EcalPseudoStripInputSample::EcalPseudoStripInputSample(uint16_t data) : theSample(data) { }

EcalPseudoStripInputSample::EcalPseudoStripInputSample(int pseudoStripInput, bool fineGrain) { 
    theSample=(pseudoStripInput&0xFFF)|((fineGrain)?(0x1000):(0));
}


std::ostream& operator<<(std::ostream& s, const EcalPseudoStripInputSample& samp) {
  return s << "PSInput=" << samp.pseudoStripInput() << ", FG=" << samp.fineGrain();
}
