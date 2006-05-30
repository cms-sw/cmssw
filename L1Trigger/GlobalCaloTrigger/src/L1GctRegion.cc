#include "L1Trigger/GlobalCaloTrigger/interface/L1GctRegion.h"

using std::ostream;
using std::endl;

L1GctRegion::L1GctRegion(int eta, int phi, unsigned et, bool mip, bool quiet, bool tauVeto, bool overFlow):
  m_mip(mip),
  m_quiet(quiet),
  m_tauVeto(tauVeto),
  m_overFlow(overFlow)
{
  setEt(et);
  setPhi(phi);
  setEt(et);
}

L1GctRegion::~L1GctRegion()
{
}

// TODO : set correct range checks
void L1GctRegion::setEta(int eta) {
  if (eta>-999 && eta<999) {
    m_eta = eta; 
  }
}

// TODO : set correct range checks
void L1GctRegion::setPhi(int phi) {
  if (phi>=0 && phi<999) {
    m_phi = phi;
  }
}

// TODO : set correct range checks
void L1GctRegion::setEt(unsigned et) {
    m_et = et & 0xffff;
} 

void L1GctRegion::setMip(bool mip) {
  m_mip = mip; 
}

void L1GctRegion::setQuiet(bool quiet) {
  m_quiet = quiet; 
}

void L1GctRegion::setTauVeto(bool tauVeto) {
  m_tauVeto = tauVeto; 
}

void L1GctRegion::setOverFlow(bool overFlow) {
  m_overFlow = overFlow; 
}


std::ostream& operator << (std::ostream& os, const L1GctRegion& reg)
{
  os << "Et " << reg.m_et;
  os << " Eta " << reg.m_eta;
  os << " Phi " << reg.m_phi;
  os << " MIP " << reg.m_mip;
  os << " QUIET " << reg.m_quiet;
  os << " TAU " << reg.m_tauVeto;
  os << " OVF " << reg.m_overFlow << endl;

  return os;
}
