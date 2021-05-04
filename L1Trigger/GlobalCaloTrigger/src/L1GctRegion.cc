#include "L1Trigger/GlobalCaloTrigger/interface/L1GctRegion.h"

L1GctRegion::L1GctRegion() : L1CaloRegion() {}

L1GctRegion::~L1GctRegion() {}

L1GctRegion L1GctRegion::makeJfInputRegion(const L1CaloRegion& r_in) {
  bool of = (r_in.isHf() ? (r_in.et() == 0xff) : r_in.overFlow());
  L1GctRegion r(r_in.et(), of, r_in.fineGrain(), r_in.gctEta(), r_in.gctPhi(), r_in.bx());
  return r;
}

L1GctRegion L1GctRegion::makeProtoJetRegion(const unsigned et,
                                            const bool overFlow,
                                            const bool fineGrain,
                                            const bool tauIsolationVeto,
                                            const unsigned ieta,
                                            const unsigned iphi,
                                            const int16_t bx) {
  L1GctRegion r(et, overFlow, fineGrain, ieta, iphi, bx);
  if (tauIsolationVeto) {
    r.setFeatureBit0();
  } else {
    r.clrFeatureBit0();
  }
  return r;
}

L1GctRegion L1GctRegion::makeFinalJetRegion(const unsigned et,
                                            const bool overFlow,
                                            const bool fineGrain,
                                            const unsigned ieta,
                                            const unsigned iphi,
                                            const int16_t bx) {
  L1GctRegion r(et, overFlow, fineGrain, ieta, iphi, bx);
  return r;
}

// constructor for internal use
L1GctRegion::L1GctRegion(const unsigned et,
                         const bool overFlow,
                         const bool fineGrain,
                         const unsigned ieta,
                         const unsigned iphi,
                         const int16_t bx)
    : L1CaloRegion(L1CaloRegion::makeGctJetRegion(
          ((overFlow || et > kGctRegionMaxValue) ? (unsigned)kGctRegionMaxValue : (unsigned)(et & kGctRegionMaxValue)),
          (overFlow || et > kGctRegionMaxValue),
          fineGrain,
          ieta,
          iphi,
          bx)) {}

void L1GctRegion::setBit(const unsigned bitNum, const bool onOff) {
  if ((bitNum == 14) || (bitNum == 15)) {
    uint16_t data = raw();
    uint16_t mask = 1 << bitNum;
    data &= ~mask;
    if (onOff)
      data |= mask;
    setRawData(data);
  }
}
