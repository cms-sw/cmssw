#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJet.h"

#include "FWCore/Utilities/interface/Exception.h"  

//DEFINE STATICS
const unsigned L1GctJet::LOCAL_ETA_HF_START = 7;
const unsigned L1GctJet::RAWSUM_BITWIDTH = L1GctJetEtCalibrationLut::JET_ENERGY_BITWIDTH;  
const unsigned L1GctJet::ETA_BITWIDTH = 4;
const unsigned L1GctJet::PHI_BITWIDTH = 5;
const unsigned L1GctJet::N_RGN_ETA = L1GctMap::N_RGN_ETA;
const unsigned L1GctJet::N_RGN_PHI = L1GctMap::N_RGN_PHI;


L1GctJet::L1GctJet(uint16_t rawsum, uint16_t eta, uint16_t phi, bool tauVeto,
		   L1GctJetEtCalibrationLut* lut) :
  m_rawsum(rawsum),
  m_eta(eta),
  m_phi(phi),
  m_tauVeto(tauVeto),
  m_jetEtCalibrationLut(lut)
{

}

L1GctJet::~L1GctJet()
{
}

std::ostream& operator << (std::ostream& os, const L1GctJet& cand)
{
  os << "L1 Gct jet";
  os << " energy sum " << cand.m_rawsum;
  os << " Eta " << cand.m_eta;
  os << " Phi " << cand.m_phi;
  os << " Tau " << cand.m_tauVeto;
  if (cand.m_jetEtCalibrationLut == 0) {
    os << " using default lut!" << std::endl;
  } else {
    os << " rank " << cand.rank();
    os << " lut address " << cand.m_jetEtCalibrationLut << std::endl;
  }

  return os;
}	

void L1GctJet::setupJet(uint16_t rawsum, uint16_t eta, uint16_t phi, bool tauVeto)
{
    m_rawsum = rawsum;
    m_eta = eta;
    m_phi = phi;
    m_tauVeto = tauVeto;    
}

/// Methods to return the jet rank
uint16_t L1GctJet::rank()      const
{
  uint16_t result;
  // If no lut setup, just return the MSB of the rawsum as the rank
  if (m_jetEtCalibrationLut==0) {
    result = std::min(63, m_rawsum >> (RAWSUM_BITWIDTH - 6));
  } else {
    result = m_jetEtCalibrationLut->convertToSixBitRank(m_rawsum, m_eta);
  }
  return result;
}

uint16_t L1GctJet::rankForHt() const
{
  uint16_t result;
  // If no lut setup, just return the MSB of the rawsum as the rank
  if (m_jetEtCalibrationLut==0) {
    result = std::min(1023, m_rawsum >> (RAWSUM_BITWIDTH - 10));
  } else {
    result = m_jetEtCalibrationLut->convertToTenBitRank(m_rawsum, m_eta);
  }
  return result;
}

/// convert to central jet digi
L1GctJetCand L1GctJet::makeJetCand() {
  return L1GctJetCand(this->rank(), this->hwEta(), this->hwPhi(), this->isTauJet(), this->isForwardJet());
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
