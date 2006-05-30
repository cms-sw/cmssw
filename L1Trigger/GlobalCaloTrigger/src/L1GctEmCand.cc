//this must be changed when scramming is working
#include"L1Trigger/GlobalCaloTrigger/interface/L1GctEmCand.h"

//DEFINE STATICS
const int L1GctEmCand::RANK_BITWIDTH = 6;
const int L1GctEmCand::ETA_BITWIDTH = 4;
const int L1GctEmCand::PHI_BITWIDTH = 5;


L1GctEmCand::L1GctEmCand(unsigned rank, unsigned eta, unsigned phi)
{
  m_rank = rank & 0x3f;
  m_eta  = eta  & 0xf;
  m_phi  = phi  & 0x1f;
}

L1GctEmCand::~L1GctEmCand(){
}
	
std::ostream& operator << (std::ostream& os, const L1GctEmCand& cand)
{
  os << "Rank " << cand.m_rank;
  os << " Eta " << cand.m_eta;
  os << " Phi " << cand.m_phi << std::endl;

  return os;
}	

/// convert to iso em digi
L1GctIsoEm L1GctEmCand::makeIsoEm() {
  return L1GctIsoEm(m_rank, m_eta, m_phi);
}

/// convert to non-iso em digi
L1GctNonIsoEm L1GctEmCand::makeNonIsoEm() {
  return L1GctNonIsoEm(m_rank, m_eta, m_phi);
}


