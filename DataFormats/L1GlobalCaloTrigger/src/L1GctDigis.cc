#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctDigis.h"

#include <iostream>

using std::ostream;
using std::endl;

L1GctCand::L1GctCand() : m_data(0) { }

L1GctCand::L1GctCand(uint16_t data) : m_data(data) { }

L1GctCand::L1GctCand(int rank, int phi, int eta) {
  m_data = (rank & 0x3f) + ((phi & 0x1f)<<6) + ((eta & 0xf)<<11);
}

L1GctCand::~L1GctCand() { } 


L1GctEmCand::L1GctEmCand() { }

L1GctEmCand::L1GctEmCand(uint16_t data) : L1GctCand(data) { }

L1GctEmCand::L1GctEmCand(int rank, int phi, int eta, bool iso, unsigned rctCrate) : 
  L1GctCand(rank, phi, eta),
  m_iso(iso),
  m_rctCrate(rctCrate) {
 }

L1GctEmCand::~L1GctEmCand() { } 


L1GctJetCand::L1GctJetCand() { }

L1GctJetCand::L1GctJetCand(uint16_t data) : L1GctCand(data) { }

L1GctJetCand::L1GctJetCand(int rank, int phi, int eta, bool isTau, bool isFor) : 
  L1GctCand(rank, phi, eta),
  m_isTau(isTau),
  m_isFor(isFor)
{ }

L1GctJetCand::~L1GctJetCand() { } 


ostream& operator<<(ostream& s, const L1GctCand& cand) {
  s << "rank=" << cand.rank() << ", eta=" << cand.eta() << ", phi=" << cand.phi();
  return s;
}

ostream& operator<<(ostream& s, const L1GctEmCand& cand) {
  s << "rank=" << cand.rank() << ", eta=" << cand.eta() << ", phi=" << cand.phi();
  s << ", iso=" << cand.isolated() << ", RCT=" << cand.rctCrate();
  return s;
}

ostream& operator<<(ostream& s, const L1GctJetCand& cand) {
  s << "rank=" << cand.rank() << ", eta=" << cand.eta() << ", phi=" << cand.phi();
  s << ", isTau=" << cand.isTau() << ", isFor=" << cand.isFor();
  return s;
}

