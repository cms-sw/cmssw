#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetCand.h"

#include <assert.h>


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
