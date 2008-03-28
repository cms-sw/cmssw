#include "L1Trigger/GlobalCaloTrigger/interface/L1GctRegion.h"

L1GctRegion::L1GctRegion(const unsigned et, const bool overFlow, const bool fineGrain,
                         const unsigned ieta, const unsigned iphi, const int16_t bx) :
  L1CaloRegion( ((et>kGctRegionMaxValue) ? kGctRegionMaxValue : et),
                ((et>kGctRegionMaxValue) || overFlow), 
                fineGrain, false, false, ieta, iphi)
{
  this->setBx(bx);
}

L1GctRegion::L1GctRegion(const unsigned et, const bool overFlow, const bool fineGrain,
                         const unsigned ieta, const unsigned iphi) :
  L1CaloRegion( ((et>kGctRegionMaxValue) ? kGctRegionMaxValue : et),
                ((et>kGctRegionMaxValue) || overFlow), 
                fineGrain, false, false, ieta, iphi)
{
  this->setBx(0);
}

L1GctRegion::L1GctRegion(const L1CaloRegion& r) :
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
