#include "L1Trigger/GlobalCaloTrigger/interface/L1GctRegion.h"

L1GctRegion::L1GctRegion(const unsigned et, const bool overFlow, const bool tauVeto, const unsigned ieta, const unsigned iphi) :
  L1CaloRegion( ((et>kGctRegionMaxValue) ? kGctRegionMaxValue : et),
                ((et>kGctRegionMaxValue) || overFlow), 
                tauVeto, false, false, ieta, iphi)
{
}

L1GctRegion::L1GctRegion(const L1CaloRegion& r) :
  L1CaloRegion( (r.overFlow() ? kGctRegionMaxValue : r.et()),
                 r.overFlow(), r.tauVeto(), false, false, r.gctEta(), r.gctPhi())
{
}

L1GctRegion::L1GctRegion() : L1CaloRegion()
{
}

L1GctRegion::~L1GctRegion()
{
}
