#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJet.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetEtCalibrationLut.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"



L1GctJet::L1GctJet(const uint16_t rawsum, const unsigned eta, const unsigned phi, const bool overFlow,
                   const bool forwardJet, const bool tauVeto, const int16_t bx) :
  m_rawsum(rawsum & kRawsumMaxValue),
  m_id(eta, phi),
  m_overFlow(overFlow || (rawsum>kRawsumMaxValue)),
  m_forwardJet(forwardJet),
  m_tauVeto(tauVeto || forwardJet),
  m_bx(bx)
{
}

L1GctJet::~L1GctJet()
{
}

std::ostream& operator << (std::ostream& os, const L1GctJet& cand)
{
  os << "L1 Gct jet";
  os << " energy sum " << cand.m_rawsum;
  if (cand.overFlow()) { os << ", overflow bit set;"; }
  os << " Eta " << cand.globalEta();
  os << " Phi " << cand.globalPhi();
  if (cand.isForwardJet()) { os << ", Forward jet"; }
  if (cand.isCentralJet()) { os << ", Central jet"; }
  if (cand.isTauJet()) { os << ", Tau jet"; }
  if (cand.isNullJet()) { os << ", Null jet"; }

  return os;
}	

/// test whether two jets are the same
bool L1GctJet::operator== (const L1GctJet& cand) const
{
  bool result=true;
  result &= (this->rawsum()==cand.rawsum());
  result &= (this->overFlow()==cand.overFlow());
  result &= (this->isForwardJet()==cand.isForwardJet());
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
  result |= !(this->overFlow()==cand.overFlow());
  result |= !(this->isForwardJet()==cand.isForwardJet());
  result |= !(this->tauVeto()==cand.tauVeto());
  result |= !(this->globalEta()==cand.globalEta());
  result |= !(this->globalPhi()==cand.globalPhi());
  result &= !(this->isNullJet() && cand.isNullJet());
  return result;
}
  
void L1GctJet::setupJet(const uint16_t rawsum, const unsigned eta, const unsigned phi, const bool overFlow,
                        const bool forwardJet, const bool tauVeto, const int16_t bx)
{
  L1CaloRegionDetId temp(eta, phi);
  m_rawsum = rawsum & kRawsumMaxValue;
  m_id = temp;
  m_overFlow = (overFlow || rawsum>kRawsumMaxValue);
  m_forwardJet = forwardJet;
  m_tauVeto = tauVeto || forwardJet;
  m_bx = bx;
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

/// Function to convert from internal format to external jet candidates at the output of the jetFinder 
L1GctJetCand L1GctJet::jetCand(const lutPtr lut) const
{
  return L1GctJetCand(rank(lut), hwPhi(), hwEta(), isTauJet(), isForwardJet(), (uint16_t) 0, (uint16_t) 0, m_bx);
}

/// Function to convert from internal format to external jet candidates at the output of the jetFinder 
L1GctJetCand L1GctJet::jetCand(const std::vector<lutPtr>& luts) const
{
  L1GctJetCand result;
  if (rctEta() < luts.size()) result = jetCand(luts.at(rctEta()));
  return result;
}

/// The two separate Lut outputs
uint16_t L1GctJet::rank(const lutPtr lut) const
{
  return lutValue(lut); 
}

unsigned L1GctJet::calibratedEt(const lutPtr lut) const
{
  return m_rawsum;
}

// internal function to find the lut contents for a jet
uint16_t L1GctJet::lutValue(const lutPtr lut) const
{
  uint16_t result; 
  if (m_overFlow) { 
    // Set output values to maximum 
    result = 0x3f; 
  } else { 
    unsigned addrBits = m_rawsum;
    // Set the MSB for tau jets
    if (!m_tauVeto && !m_forwardJet) {
      addrBits |= 1 << (L1GctJetEtCalibrationLut::JET_ENERGY_BITWIDTH);
    }
    uint16_t address = static_cast<uint16_t>(addrBits);
    result = lut->lutValue(address);
  }
  return result;
}
