#include "L1Trigger/GlobalCaloTrigger/interface/L1GctRegion.h"

L1GctRegion::L1GctRegion() : L1CaloRegion() {}

L1GctRegion::L1GctRegion(const L1CaloRegion& r) :
  L1CaloRegion( L1CaloRegion::makeGctJetRegion( (r.overFlow() ? kGctRegionMaxValue : r.et()),
						r.overFlow(), r.fineGrain(), r.gctEta(), r.gctPhi(), r.bx()) ) {}

L1GctRegion::~L1GctRegion()
{
}

L1GctRegion L1GctRegion::makeProtoJetRegion(const unsigned et, const bool overFlow, const bool fineGrain, const bool tauIsolationVeto,
                                            const unsigned ieta, const unsigned iphi, const int16_t bx)
{
  L1GctRegion r( L1CaloRegion::makeGctJetRegion( (((et>kGctRegionMaxValue) || overFlow) ? kGctRegionMaxValue : et),
						 ((et>kGctRegionMaxValue) || overFlow), fineGrain, ieta, iphi, bx) );
  if (tauIsolationVeto) {
    r.setFeatureBit0();
  } else {
    r.clrFeatureBit0();
  }
  return r;
}

L1GctRegion L1GctRegion::makeFinalJetRegion(const unsigned et, const bool overFlow, const bool fineGrain,
                                            const unsigned ieta, const unsigned iphi, const int16_t bx)
{
  L1GctRegion r( L1CaloRegion::makeGctJetRegion( (((et>kGctRegionMaxValue) || overFlow) ? kGctRegionMaxValue : et),
						 ((et>kGctRegionMaxValue) || overFlow), fineGrain, ieta, iphi, bx) );
  return r;
}

void L1GctRegion::setBit(const unsigned bitNum, const bool onOff)
{
  if ((bitNum==14) || (bitNum==15)) { 
    uint16_t data = raw();
    uint16_t mask = 1 << bitNum;
    data &= !mask;
    if (onOff) data |= mask;
    setRawData(data);
  } 
}
