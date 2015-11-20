#ifndef MuonRawDigiTranslator_h
#define MuonRawDigiTranslator_h

#include "DataFormats/L1Trigger/interface/Muon.h"

namespace l1t {
  class MuonRawDigiTranslator {
    public:
      static void fillMuon(Muon&, uint32_t, uint32_t);
      static void fillMuon(Muon&, uint64_t);
      static void generatePackedDataWords(const Muon&, uint32_t&, uint32_t&);
      static uint64_t generate64bitDataWord(const Muon&);

      static const unsigned ptMask_ = 0x1FF;
      static const unsigned ptShift_ = 10;
      static const unsigned qualMask_ = 0xF;
      static const unsigned qualShift_ = 19;
      static const unsigned absEtaMask_ = 0xFF;
      static const unsigned absEtaShift_ = 23;
      static const unsigned etaSignShift_ = 31;
      static const unsigned phiMask_ = 0x3FF;
      static const unsigned phiShift_ = 0;
      static const unsigned chargeShift_ = 2;
      static const unsigned chargeValidShift_ = 3;
      static const unsigned isoMask_ = 0x3;
      static const unsigned isoShift_ = 0;
  };
}

#endif
