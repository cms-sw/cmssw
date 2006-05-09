#include "L1Trigger/GlobalCaloTrigger/interface/L1GctRegion.h"

using namespace std;


L1GctRegion::L1GctRegion(int eta, int phi, ULong et, bool mip, bool quiet, bool tauVeto, bool overFlow):
    m_eta(eta),
    m_phi(phi),
    myEt(et),
    myMip(mip),
    myQuiet(quiet),
    myTauVeto(tauVeto),
    myOverFlow(overFlow)
{
}

L1GctRegion::~L1GctRegion()
{
}
