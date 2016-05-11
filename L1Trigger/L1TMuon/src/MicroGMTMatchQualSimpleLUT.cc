#include "../interface/MicroGMTMatchQualLUT.h"
#include "TMath.h"

l1t::MicroGMTMatchQualSimpleLUT::MicroGMTMatchQualSimpleLUT (const std::string& fname, const double maxDR, const double fEta, const double fPhi, cancel_t cancelType) : MicroGMTMatchQualLUT()
{
  m_dEtaRedInWidth = 5;
  m_dPhiRedInWidth = 3;
  m_maxDR = maxDR;
  m_fEta = fEta;
  m_fPhi = fPhi;
  m_cancelType = cancelType;
  
  m_totalInWidth = m_dPhiRedInWidth + m_dEtaRedInWidth;
  m_outWidth = 1;

  m_dPhiRedMask = (1 << m_dPhiRedInWidth) - 1;
  m_dEtaRedMask = ((1 << m_dEtaRedInWidth) - 1) << m_dPhiRedInWidth;

  m_inputs.push_back(MicroGMTConfiguration::DELTA_ETA_RED);
  m_inputs.push_back(MicroGMTConfiguration::DELTA_PHI_RED);

  m_phiScale = 2*TMath::Pi()/576.0;
  m_etaScale = 0.010875;

  if (fname != std::string("")) {
    load(fname);
  } else {
    initialize();
  }
}

l1t::MicroGMTMatchQualSimpleLUT::MicroGMTMatchQualSimpleLUT (l1t::LUT* lut, cancel_t cancelType) : MicroGMTMatchQualLUT(lut)
{
  m_dEtaRedInWidth = 5;
  m_dPhiRedInWidth = 3;
  m_cancelType = cancelType;

  m_totalInWidth = m_dPhiRedInWidth + m_dEtaRedInWidth;
  m_outWidth = 1;

  m_dPhiRedMask = (1 << m_dPhiRedInWidth) - 1;
  m_dEtaRedMask = ((1 << m_dEtaRedInWidth) - 1) << m_dPhiRedInWidth;

  m_inputs.push_back(MicroGMTConfiguration::DELTA_ETA_RED);
  m_inputs.push_back(MicroGMTConfiguration::DELTA_PHI_RED);

  m_phiScale = 2*TMath::Pi()/576.0;
  m_etaScale = 0.010875;

  m_initialized = true;
}

int
l1t::MicroGMTMatchQualSimpleLUT::lookup(int etaFine, int dEtaRed, int dPhiRed) const // etaFine will be ignored
{
  // normalize these two to the same scale and then calculate?
  if (m_initialized) {
    return data((unsigned)hashInput(checkedInput(dEtaRed, m_dEtaRedInWidth), checkedInput(dPhiRed, m_dPhiRedInWidth)));
  }
  double dEta = m_fEta*dEtaRed*m_etaScale;
  double dPhi = m_fPhi*dPhiRed*m_phiScale;
  double dR = std::sqrt(dEta*dEta + dPhi*dPhi);

  int retVal = dR <= m_maxDR ? 1 : 0;

  return retVal;
}

int
l1t::MicroGMTMatchQualSimpleLUT::lookupPacked(int in) const
{
  if (m_initialized) {
    return data((unsigned)in);
  }

  int dEtaRed = 0;
  int dPhiRed = 0;
  unHashInput(in, dEtaRed, dPhiRed);
  return lookup(0, dEtaRed, dPhiRed);
}

int
l1t::MicroGMTMatchQualSimpleLUT::hashInput(int dEtaRed, int dPhiRed) const
{

  int result = 0;
  result += dPhiRed;
  result += dEtaRed << m_dPhiRedInWidth;
  return result;
}

void
l1t::MicroGMTMatchQualSimpleLUT::unHashInput(int input, int& dEtaRed, int& dPhiRed) const
{
  dPhiRed = input & m_dPhiRedMask;
  dEtaRed = (input & m_dEtaRedMask) >> m_dPhiRedInWidth;
}
