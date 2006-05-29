#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetCand.h"

#include <assert.h>

//DEFINE STATICS
const int L1GctJetCand::LOCAL_ETA_HF_START = 7;
const int L1GctJetCand::RANK_BITWIDTH = 6;  
const int L1GctJetCand::ETA_BITWIDTH = 4;
const int L1GctJetCand::PHI_BITWIDTH = 5;


L1GctJetCand::L1GctJetCand(uint16_t rank, uint16_t eta, uint16_t phi, bool tauVeto) :
  m_rank(rank),
  m_eta(eta),
  m_phi(phi),
  m_tauVeto(tauVeto)
{

}

L1GctJetCand::~L1GctJetCand()
{
}

std::ostream& operator << (std::ostream& os, const L1GctJetCand& cand)
{
  os << "Rank " << cand.m_rank;
  os << " Eta " << cand.m_eta;
  os << " Phi " << cand.m_phi;
  os << " Tau " << cand.m_tauVeto << std::endl;

  return os;
}	

void L1GctJetCand::setupJet(uint16_t rank, uint16_t eta, uint16_t phi, bool tauVeto)
{
    m_rank = rank;
    m_eta = eta;
    m_phi = phi;
    m_tauVeto = tauVeto;    
}

L1GctJetCand L1GctJetCand::convertToGlobalJet(int jetFinderPhiIndex, int wheelId)
{
    //Some debug checks...
    assert(jetFinderPhiIndex >= 0 && jetFinderPhiIndex < 9);
    assert(wheelId == 0 || wheelId == 1);
    assert(m_eta < 11);  //Eta should run from 0 to 10 in local jetfinder co-ords
    assert(m_phi < 2);  //Phi should be either 0 or 1 in local jetfinder co-ords

    L1GctJetCand outputJet = *this;  //copy this instance to a temporary jet.

    //remove the ability to distinguish between central and forward jets
    if(m_eta >= LOCAL_ETA_HF_START) { outputJet.setRank(m_rank - LOCAL_ETA_HF_START); }
    
    //the MSB of the eta address must be set to 1, to show -ve co-ord. 
    if(wheelId == 0) { outputJet.setEta(m_eta + (1 << (ETA_BITWIDTH-1))); }

    outputJet.setPhi(m_phi + jetFinderPhiIndex*2);
    
    return outputJet;    
}

/// convert to central jet digi
L1GctCenJet L1GctJetCand::makeCenJet() {
  /// TODO : include check that this jet really *is* central ?
  return L1GctCenJet(m_rank, m_eta, m_phi);
}

/// convert to forward jet digi
L1GctForJet L1GctJetCand::makeForJet() {
  /// TODO : include check that this jet really *is* forward ?
  return L1GctForJet(m_rank, m_eta, m_phi);
}

/// convert to tau jet digi
L1GctTauJet L1GctJetCand::makeTauJet() {
  /// TODO : include check that this jet really *is* tau ?
  return L1GctTauJet(m_rank, m_eta, m_phi);
}
