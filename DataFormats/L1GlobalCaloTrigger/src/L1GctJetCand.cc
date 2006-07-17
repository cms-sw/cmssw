

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"

#include <ostream>

using std::ostream;

L1GctJetCand::L1GctJetCand() :
  m_data(0),
  m_isTau(false),
  m_isFor(false)
{

}

L1GctJetCand::L1GctJetCand(uint16_t data, bool isTau, bool isFor) : 
  m_data(data),
  m_isTau(isTau),
  m_isFor(isFor)
{
}

L1GctJetCand::L1GctJetCand(unsigned rank, int phi, int eta, bool isTau, bool isFor) : 
  m_isTau(isTau),
  m_isFor(isFor)
{ 
  m_data = (rank & 0x3f) + 
    ((static_cast<unsigned>(eta) & 0x7)<<6) +
    ((static_cast<unsigned>(phi) & 0xf)<<10); 
}

L1GctJetCand::~L1GctJetCand() { } 

// pretty print
ostream& operator<<(ostream& s, const L1GctJetCand& cand) {
  s << "L1GctJetCand : ";
  s << "rank=" << cand.rank();
  s << ", eta=" << cand.level1EtaIndex() << ", phi=" << cand.level1PhiIndex();
  s << " type=";
  if (cand.isTau()) { s << "tau"; }
  else if (cand.isForward()) { s << "forward"; }
  else { s << "central"; }
  return s;
}
