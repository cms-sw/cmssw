#ifndef MuonRawDigiTranslator_h
#define MuonRawDigiTranslator_h

#include "DataFormats/L1Trigger/interface/Muon.h"

namespace l1t {
  class MuonRawDigiTranslator {
  public:
    static void fillMuon(Muon&, uint32_t, uint32_t, int, unsigned int);
    static void fillMuon(Muon&, uint64_t, int, unsigned int);
    static void generatePackedDataWords(const Muon&, uint32_t&, uint32_t&);
    static uint64_t generate64bitDataWord(const Muon&);
    static int calcHwEta(const uint32_t&, const unsigned, const unsigned);

    static const unsigned ptMask_ = 0x1FF;
    static const unsigned ptShift_ = 10;
    static const unsigned qualMask_ = 0xF;
    static const unsigned qualShift_ = 19;
    static const unsigned absEtaMask_ = 0xFF;
    static const unsigned absEtaShift_ = 21;
    static const unsigned absEtaAtVtxShift_ = 23;
    static const unsigned etaSignShift_ = 29;
    static const unsigned etaAtVtxSignShift_ = 31;
    static const unsigned phiMask_ = 0x3FF;
    static const unsigned phiShift_ = 11;
    static const unsigned phiAtVtxShift_ = 0;
    static const unsigned chargeShift_ = 2;
    static const unsigned chargeValidShift_ = 3;
    static const unsigned tfMuonIndexMask_ = 0x7F;
    static const unsigned tfMuonIndexShift_ = 4;
    static const unsigned isoMask_ = 0x3;
    static const unsigned isoShift_ = 0;
  };
}  // namespace l1t

#endif
