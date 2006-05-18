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

L1GctRegion::L1GctRegion(ULong rawData)
{

    myEt = rawData & 0x3f;  //will put the first 10 bits of rawData into the Et
    
    rawData >>= ET_BITWIDTH;  //shift the remaining bits down to remove the 10 bits of Et
    
    myOverFlow = rawData & 0x1; //LSB is now overflow bit
    myTauVeto = (rawData & 0x2) >> 1; //2nd bit is tauveto
    
}

L1GctRegion::~L1GctRegion()
{
}

std::ostream& operator << (std::ostream& os, const L1GctRegion& reg)
{
  os << "Et " << reg.myEt;
  os << " Eta " << reg.m_eta;
  os << " Phi " << reg.m_phi;
  os << " MIP " << reg.myMip;
  os << " QUIET " << reg.myQuiet;
  os << " TAU " << reg.myTauVeto;
  os << " OVF " << reg.myOverFlow << std::endl;

  return os;
}
