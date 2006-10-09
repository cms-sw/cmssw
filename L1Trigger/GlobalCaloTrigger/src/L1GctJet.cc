#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJet.h"

#include "FWCore/Utilities/interface/Exception.h"  

//DEFINE STATICS
const unsigned L1GctJet::RAWSUM_BITWIDTH = 10;


L1GctJet::L1GctJet(uint16_t rawsum, unsigned eta, unsigned phi, bool tauVeto,
		   L1GctJetEtCalibrationLut* lut) :
  m_rawsum(rawsum),
  m_id(eta, phi),
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
  os << " Eta " << cand.globalEta();
  os << " Phi " << cand.globalPhi();
  os << " Tau " << cand.m_tauVeto;
  os << " rank " << cand.rank();
  if (cand.m_jetEtCalibrationLut == 0) {
    os << " using default lut!";
  } else {
    os << " lut address " << cand.m_jetEtCalibrationLut;
  }

  return os;
}	

/// test whether two jets are the same
bool L1GctJet::operator== (const L1GctJet& cand) const
{
  bool result=true;
  result &= (this->rawsum()==cand.rawsum());
  result &= (this->tauVeto()==cand.tauVeto());
  result &= (this->globalEta()==cand.globalEta());
  result &= (this->globalPhi()==cand.globalPhi());
  result |= (this->isNullJet() && cand.isNullJet());
  return result;
}
  
/// test whether two jets are different
bool L1GctJet::operator!= (const L1GctJet& cand) const
{
  bool result=false;
  result |= !(this->rawsum()==cand.rawsum());
  result |= !(this->tauVeto()==cand.tauVeto());
  result |= !(this->globalEta()==cand.globalEta());
  result |= !(this->globalPhi()==cand.globalPhi());
  result &= !(this->isNullJet() && cand.isNullJet());
  return result;
}
  
void L1GctJet::setupJet(uint16_t rawsum, unsigned eta, unsigned phi, bool tauVeto)
{
  L1CaloRegionDetId temp(eta, phi);
  m_rawsum = rawsum;
  m_id = temp;
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
    result = m_jetEtCalibrationLut->rank(m_rawsum, m_id.ieta());
  }
  return result;
}

uint16_t L1GctJet::calibratedEt() const
{
  uint16_t result;
  // If no lut setup, just return the MSB of the rawsum as the rank
  if (m_jetEtCalibrationLut==0) {
    result = std::min(1023, m_rawsum >> (RAWSUM_BITWIDTH - 10));
  } else {
    result = m_jetEtCalibrationLut->calibratedEt(m_rawsum, m_id.ieta());
  }
  return result;
}

/// convert to central jet digi
L1GctJetCand L1GctJet::makeJetCand() {
  return L1GctJetCand(this->rank(), this->hwPhi(), this->hwEta(), this->isTauJet(), this->isForwardJet());
}

/// eta value as encoded in hardware at the GCT output
unsigned L1GctJet::hwEta() const
{
  // Force into 4 bits.
  // Count eta bins separately for central and forward jets. Set MSB to indicate the Wheel
  return (((m_id.rctEta() % 7) & 0x7) | (m_id.ieta()<11 ? 0x8 : 0));
}

/// phi value as encoded in hardware at the GCT output
unsigned L1GctJet::hwPhi() const
{
  // Force into 5 bits.
  return m_id.iphi() & 0x1f;
}
