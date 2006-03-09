#include "../interface/L1GctRegion.h"

L1GctRegion::L1GctRegion()
{
}

L1GctRegion::L1GctRegion(ULong et, bool mip, bool quiet):
m_mip(mip),
m_quiet(quiet)
{
    TenBit tempEt(et);
    m_et = tempEt;
}

L1GctRegion::~L1GctRegion()
{
}
