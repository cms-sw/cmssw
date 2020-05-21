#include "RecoLocalMuon/CSCRecHitD/src/CSCStripHit.h"
#include <iostream>

CSCStripHit::CSCStripHit(const CSCDetId& id,
                         const float& sHitPos,
                         const int& tmax,
                         const ChannelContainer& strips,
                         const StripHitADCContainer& s_adc,
                         const StripHitADCContainer& s_adcRaw,
                         const int& numberOfConsecutiveStrips,
                         const int& closestMaximum,
                         const short int& deadStrip)
    : theDetId(id),
      theStripHitPosition(sHitPos),
      theStripHitTmax(tmax),
      theStrips(strips),
      theStripHitADCs(s_adc),
      theStripHitRawADCs(s_adcRaw),
      theConsecutiveStrips(numberOfConsecutiveStrips),
      theClosestMaximum(closestMaximum),
      theDeadStrip(deadStrip) {
  /// Extract the 2 lowest bytes for strip number
  theStripsLowBits.clear();
  for (int theStrip : theStrips) {
    theStripsLowBits.push_back(theStrip & 0x000000FF);
  }
  /// Extract the 2 highest bytes for L1A phase
  theStripsHighBits.clear();
  for (int theStrip : theStrips) {
    theStripsHighBits.push_back(theStrip & 0x0000FF00);
  }
}

/// Debug
void CSCStripHit::print() const {
  std::cout << "CSCStripHit in CSC Detector: " << std::dec << cscDetId() << std::endl;
  std::cout << "  sHitPos: " << sHitPos() << std::endl;
  std::cout << "  TMAX: " << tmax() << std::endl;
  std::cout << "  STRIPS: ";
  for (int i : strips()) {
    std::cout << std::dec << i << " ("
              << "HEX: " << std::hex << i << ")"
              << " ";
  }
  std::cout << std::endl;

  /// L1A
  std::cout << "  L1APhase: ";
  for (int i : stripsl1a()) {
    //uint16_t L1ABitSet=(strips()[i] & 0xFF00);
    //std::cout << std::hex << (stripsl1a()[i] >> 15)<< " ";

    std::cout << "|";
    for (int k = 0; k < 8; k++) {
      std::cout << ((i >> (15 - k)) & 0x1) << " ";
    }
    std::cout << "| ";
  }
  std::cout << std::endl;

  std::cout << "  S_ADC: ";
  for (float i : s_adc()) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  std::cout << "  S_ADC_RAW: ";
  for (float i : s_adcRaw()) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
}
