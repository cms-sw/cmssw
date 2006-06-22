#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJet.h"

#include <assert.h>

//DEFINE STATICS
const unsigned L1GctJet::LOCAL_ETA_HF_START = 7;
const unsigned L1GctJet::RANK_BITWIDTH = 6;  
const unsigned L1GctJet::ETA_BITWIDTH = 4;
const unsigned L1GctJet::PHI_BITWIDTH = 5;
const unsigned L1GctJet::N_RGN_ETA = L1GctMap::N_RGN_ETA;
const unsigned L1GctJet::N_RGN_PHI = L1GctMap::N_RGN_PHI;


L1GctJet::L1GctJet(uint16_t rank, uint16_t eta, uint16_t phi, bool tauVeto) :
  m_rank(rank),
  m_eta(eta),
  m_phi(phi),
  m_tauVeto(tauVeto)
{

}

L1GctJet::~L1GctJet()
{
}

std::ostream& operator << (std::ostream& os, const L1GctJet& cand)
{
  os << "Rank " << cand.m_rank;
  os << " Eta " << cand.m_eta;
  os << " Phi " << cand.m_phi;
  os << " Tau " << cand.m_tauVeto << std::endl;

  return os;
}	

void L1GctJet::setupJet(uint16_t rank, uint16_t eta, uint16_t phi, bool tauVeto)
{
    m_rank = rank;
    m_eta = eta;
    m_phi = phi;
    m_tauVeto = tauVeto;    
}

// L1GctJet L1GctJet::convertToGlobalJet(int jetFinderPhiIndex, int wheelId)
// {
//     //Some debug checks...
//     assert(jetFinderPhiIndex >= 0 && jetFinderPhiIndex < 9);
//     assert(wheelId == 0 || wheelId == 1);
//     assert(m_eta < 11);  //Eta should run from 0 to 10 in local jetfinder co-ords
//     assert(m_phi < 2);  //Phi should be either 0 or 1 in local jetfinder co-ords

//     L1GctJet outputJet = *this;  //copy this instance to a temporary jet.

//     //remove the ability to distinguish between central and forward jets
//     if(m_eta >= LOCAL_ETA_HF_START) { outputJet.setRank(m_rank - LOCAL_ETA_HF_START); }
    
//     //the MSB of the eta address must be set to 1, to show -ve co-ord. 
//     if(wheelId == 0) { outputJet.setEta(m_eta + (1 << (ETA_BITWIDTH-1))); }

//     outputJet.setPhi(m_phi + jetFinderPhiIndex*2);
    
//     return outputJet;    
// }

/// convert to central jet digi
L1GctJetCand L1GctJet::makeJetCand() {
  /// TODO : set forward bit correctly!
  return L1GctJetCand(m_rank, this->hwEta(), this->hwPhi(), this->isTauJet(), this->isForwardJet());
}

/// eta value in local jetFinder coordinates
unsigned L1GctJet::jfLocalEta() const
{
  return (m_eta < (N_RGN_ETA/2) ? ((N_RGN_ETA/2)-1-m_eta) : m_eta-(N_RGN_ETA/2) ) ;
}

/// phi value in local jetFinder coordinates
unsigned L1GctJet::jfLocalPhi() const
{
  return m_phi % 2;
}

/// eta value as encoded in hardware at the GCT output
unsigned L1GctJet::hwEta() const
{
  // Force into ETA_BITWIDTH bits. It should fit OK if things are properly set up.
  // Count eta bins separately for central and forward jets. Set MSB to indicate the Wheel
  return (((this->jfLocalEta() % LOCAL_ETA_HF_START) && ((1<<(ETA_BITWIDTH-1))-1)) || ((1-this->jfWheelIndex())<<ETA_BITWIDTH));
}

/// phi value as encoded in hardware at the GCT output
unsigned L1GctJet::hwPhi() const
{
  // Force into PHI_BITWIDTH bits. It should fit OK if things are properly set up.
  return m_phi && ((1<<PHI_BITWIDTH)-1);
}

/// the jetFinder that produced this jet
unsigned L1GctJet::jfIdNum() const
{
  return (((N_RGN_PHI + 4 - m_phi) % N_RGN_PHI) / 2) + ((m_eta/(N_RGN_ETA/2)) * (N_RGN_PHI/2));
}
