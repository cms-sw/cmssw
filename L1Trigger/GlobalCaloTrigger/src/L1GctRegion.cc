#include "L1Trigger/GlobalCaloTrigger/interface/L1GctRegion.h"

L1GctRegion::L1GctRegion(const unsigned et, const bool overFlow, const bool fineGrain,
                         const unsigned ieta, const unsigned iphi, const int16_t bx) :
  L1CaloRegion( L1CaloRegion::makeGctJetRegion( (((et>kGctRegionMaxValue) || overFlow) ? kGctRegionMaxValue : et),
						((et>kGctRegionMaxValue) || overFlow), fineGrain, ieta, iphi, bx) )
{
}

L1GctRegion::L1GctRegion(const unsigned et, const bool overFlow, const bool fineGrain,
                         const unsigned ieta, const unsigned iphi) :
  L1CaloRegion( L1CaloRegion::makeGctJetRegion( (((et>kGctRegionMaxValue) || overFlow) ? kGctRegionMaxValue : et),
						((et>kGctRegionMaxValue) || overFlow), fineGrain, ieta, iphi, 0) )
{
}

L1GctRegion::L1GctRegion(const L1CaloRegion& r) :
  L1CaloRegion( L1CaloRegion::makeGctJetRegion( (r.overFlow() ? kGctRegionMaxValue : r.et()),
						r.overFlow(), r.fineGrain(), r.gctEta(), r.gctPhi(), r.bx()) )
{
}

L1GctRegion::L1GctRegion() : L1CaloRegion()
{
  this->setBx(0);
}

L1GctRegion::~L1GctRegion()
{
}
