//this must be changed when scramming is working
#include"L1Trigger/GlobalCaloTrigger/interface/L1GctEmCand.h"

//DEFINE STATICS
const int L1GctEmCand::RANK_BITWIDTH = 6;
const int L1GctEmCand::ETA_BITWIDTH = 4;
const int L1GctEmCand::PHI_BITWIDTH = 5;

L1GctEmCand::L1GctEmCand(ULong rank, ULong eta, ULong phi) : 
  m_rank(rank),
  m_eta(eta),
  m_phi(phi)
{

}

L1GctEmCand::L1GctEmCand(ULong rawData) {
    
    m_rank = rawData & 0x3f;
    rawData >>= RANK_BITWIDTH;   //shift the remaining bits down, to remove the rank info         
    m_phi = rawData & 0x1;  //1 bit of Phi
    m_eta = (rawData & 0xE) >> 1;  //other 3 bits are eta
  
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


