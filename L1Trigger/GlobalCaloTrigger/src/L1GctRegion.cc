#include "L1Trigger/GlobalCaloTrigger/interface/L1GctRegion.h"

L1GctRegion::L1GctRegion(const unsigned et, const bool overFlow, const bool fineGrain,
                         const unsigned ieta, const unsigned iphi, const int16_t bx) :
  //   L1CaloRegion( L1CaloRegion::makeGctJetRegion( (((et>kGctRegionMaxValue) || overFlow) ? kGctRegionMaxValue : et),
    // 						((et>kGctRegionMaxValue) || overFlow), fineGrain, ieta, iphi, bx) )
// For 21X compatibility, use old-style L1CaloRegion constructor.
// For 30X, the bx argument can be included in the ctor so no need to setBx() explicitly
  L1CaloRegion( (((et>kGctRegionMaxValue) || overFlow) ? kGctRegionMaxValue : et),
		((et>kGctRegionMaxValue) || overFlow), fineGrain, false, false, ieta, iphi)
{
  this->setBx(bx);
}

L1GctRegion::L1GctRegion(const unsigned et, const bool overFlow, const bool fineGrain,
                         const unsigned ieta, const unsigned iphi) :
//   L1CaloRegion( L1CaloRegion::makeGctJetRegion( (((et>kGctRegionMaxValue) || overFlow) ? kGctRegionMaxValue : et),
// 						((et>kGctRegionMaxValue) || overFlow), fineGrain, ieta, iphi, 0) )
  L1CaloRegion( (((et>kGctRegionMaxValue) || overFlow) ? kGctRegionMaxValue : et),
		((et>kGctRegionMaxValue) || overFlow), fineGrain, false, false, ieta, iphi)
{
  this->setBx(0);
}

L1GctRegion::L1GctRegion(const L1CaloRegion& r) :
//   L1CaloRegion( L1CaloRegion::makeGctJetRegion( (r.overFlow() ? kGctRegionMaxValue : r.et()),
// 						r.overFlow(), r.fineGrain(), r.gctEta(), r.gctPhi(), r.bx()) )
  L1CaloRegion( (r.overFlow() ? kGctRegionMaxValue : r.et()),
		r.overFlow(), r.fineGrain(), false, false, r.gctEta(), r.gctPhi())
{
  this->setBx(r.bx());
}

L1GctRegion::L1GctRegion() : L1CaloRegion()
{
  this->setBx(0);
}

L1GctRegion::~L1GctRegion()
{
}
