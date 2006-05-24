#include "L1Trigger/GlobalCaloTrigger/interface/L1GctRegion.h"

using namespace std;


L1GctRegion::L1GctRegion(int eta, int phi, unsigned long et, bool mip, bool quiet, bool tauVeto, bool overFlow):
  m_eta(eta),
  m_phi(phi),
  m_et(et),
  m_mip(mip),
  m_quiet(quiet),
  m_tauVeto(tauVeto),
  m_overFlow(overFlow)
{
}

L1GctRegion::L1GctRegion(unsigned long rawData):
  m_eta(0),
  m_phi(0),
  m_mip(false),
  m_quiet(false)
{
  m_et = rawData & 0x3ff;  //will put the first 10 bits of rawData into the Et

  rawData >>= ET_BITWIDTH;  //shift the remaining bits down to remove the 10 bits of Et

  m_overFlow = rawData & 0x1; //LSB is now overflow bit
  m_tauVeto = (rawData & 0x2) >> 1; //2nd bit is tauveto
}

L1GctRegion::~L1GctRegion()
{
}

std::ostream& operator << (std::ostream& os, const L1GctRegion& reg)
{
  os << "Et " << reg.m_et;
  os << " Eta " << reg.m_eta;
  os << " Phi " << reg.m_phi;
  os << " MIP " << reg.m_mip;
  os << " QUIET " << reg.m_quiet;
  os << " TAU " << reg.m_tauVeto;
  os << " OVF " << reg.m_overFlow << std::endl;

  return os;
}
