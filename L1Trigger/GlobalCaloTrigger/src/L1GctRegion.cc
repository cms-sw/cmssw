#include "L1Trigger/GlobalCaloTrigger/interface/L1GctRegion.h"


L1GctRegion::L1GctRegion(ULong et, bool mip, bool quiet, bool tauVeto):
myEt(et),
myMip(mip),
myQuiet(quiet),
myTauVeto(tauVeto)
{
}

L1GctRegion::~L1GctRegion()
{
}
