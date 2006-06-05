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


L1GctCenJet::L1GctCenJet() { }

L1GctCenJet::L1GctCenJet(uint16_t data) : L1GctCand(data) { }

L1GctCenJet::L1GctCenJet(int rank, int phi, int eta) : L1GctCand(rank, phi, eta) { }

L1GctCenJet::~L1GctCenJet() { } 


L1GctForJet::L1GctForJet() { }

L1GctForJet::L1GctForJet(uint16_t data) : L1GctCand(data) { }

L1GctForJet::L1GctForJet(int rank, int phi, int eta) : L1GctCand(rank, phi, eta) { }

L1GctForJet::~L1GctForJet() { } 


L1GctTauJet::L1GctTauJet() { }

L1GctTauJet::L1GctTauJet(uint16_t data) : L1GctCand(data) { }

L1GctTauJet::L1GctTauJet(int rank, int phi, int eta) : L1GctCand(rank, phi, eta) { }

L1GctTauJet::~L1GctTauJet() { } 


ostream& operator<<(ostream& s, const L1GctCand& cand) {
  s << "rank=" << cand.rank() << ", eta=" << cand.eta() << ", phi=" << cand.phi();
  return s;
}

ostream& operator<<(ostream& s, const L1GctEmCand& cand) {
  s << "rank=" << cand.rank() << ", eta=" << cand.eta() << ", phi=" << cand.phi();
  return s;
}

ostream& operator<<(ostream& s, const L1GctCenJet& cand) {
  s << "rank=" << cand.rank() << ", eta=" << cand.eta() << ", phi=" << cand.phi();
  return s;
}

ostream& operator<<(ostream& s, const L1GctForJet& cand) {
  s << "rank=" << cand.rank() << ", eta=" << cand.eta() << ", phi=" << cand.phi();
  return s;
}

ostream& operator<<(ostream& s, const L1GctTauJet& cand) {
  s << "rank=" << cand.rank() << ", eta=" << cand.eta() << ", phi=" << cand.phi();
  return s;
}
