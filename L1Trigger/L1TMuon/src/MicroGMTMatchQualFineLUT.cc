#include "../interface/MicroGMTMatchQualLUT.h"
#include "TMath.h"

l1t::MicroGMTMatchQualFineLUT::MicroGMTMatchQualFineLUT (const std::string& fname, const double maxDR, const double fEta, const double fEtaCoarse, const double fPhi, cancel_t cancelType) : MicroGMTMatchQualLUT(), m_fEtaCoarse(fEtaCoarse)
{
  m_dEtaRedInWidth = 5;
  m_dPhiRedInWidth = 3;
  m_maxDR = maxDR;
  m_fEta = fEta;
  m_fPhi = fPhi;
  m_cancelType = cancelType;
  
  m_totalInWidth = 1 + m_dPhiRedInWidth + m_dEtaRedInWidth;
  m_outWidth = 1;

  m_dPhiRedMask = (1 << m_dPhiRedInWidth) - 1;
  m_dEtaRedMask = ((1 << m_dEtaRedInWidth) - 1) << m_dPhiRedInWidth;
  m_etaFineMask = 1 << (m_dEtaRedInWidth + m_dPhiRedInWidth);

  m_inputs.push_back(MicroGMTConfiguration::ETA_FINE_BIT);
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

l1t::MicroGMTMatchQualFineLUT::MicroGMTMatchQualFineLUT (l1t::LUT* lut, cancel_t cancelType) : MicroGMTMatchQualLUT(lut)
{
  m_dEtaRedInWidth = 5;
  m_dPhiRedInWidth = 3;
  m_cancelType = cancelType;

  m_totalInWidth = 1 + m_dPhiRedInWidth + m_dEtaRedInWidth;
  m_outWidth = 1;

  m_dPhiRedMask = (1 << m_dPhiRedInWidth) - 1;
  m_dEtaRedMask = ((1 << m_dEtaRedInWidth) - 1) << m_dPhiRedInWidth;
  m_etaFineMask = 1 << (m_dEtaRedInWidth + m_dPhiRedInWidth);

  m_inputs.push_back(MicroGMTConfiguration::ETA_FINE_BIT);
  m_inputs.push_back(MicroGMTConfiguration::DELTA_ETA_RED);
  m_inputs.push_back(MicroGMTConfiguration::DELTA_PHI_RED);

  m_phiScale = 2*TMath::Pi()/576.0;
  m_etaScale = 0.010875;

  m_initialized = true;
}

int
l1t::MicroGMTMatchQualFineLUT::lookup(int etaFine, int dEtaRed, int dPhiRed) const
{
  // normalize these two to the same scale and then calculate?
  if (m_initialized) {
    return data((unsigned)hashInput(checkedInput(etaFine, 1), checkedInput(dEtaRed, m_dEtaRedInWidth), checkedInput(dPhiRed, m_dPhiRedInWidth)));
  }
  double dEta = m_fEtaCoarse*dEtaRed*m_etaScale;
  if (etaFine > 0) {
    dEta = m_fEta*dEtaRed*m_etaScale;
  }
  double dPhi = m_fPhi*dPhiRed*m_phiScale;
  double dR = std::sqrt(dEta*dEta + dPhi*dPhi);

  int retVal = dR <= m_maxDR ? 1 : 0;

  return retVal;
}

int
l1t::MicroGMTMatchQualFineLUT::lookupPacked(int in) const
{
  if (m_initialized) {
    return data((unsigned)in);
  }

  int etaFine = 0;
  int dEtaRed = 0;
  int dPhiRed = 0;
  unHashInput(in, etaFine, dEtaRed, dPhiRed);
  return lookup(etaFine, dEtaRed, dPhiRed);
}

int
l1t::MicroGMTMatchQualFineLUT::hashInput(int etaFine, int dEtaRed, int dPhiRed) const
{
  int result = 0;
  result += dPhiRed;
  result += dEtaRed << m_dPhiRedInWidth;
  result += etaFine << (m_dEtaRedInWidth + m_dPhiRedInWidth);
  return result;
}

void
l1t::MicroGMTMatchQualFineLUT::unHashInput(int input, int& etaFine, int& dEtaRed, int& dPhiRed) const
{
  dPhiRed = input & m_dPhiRedMask;
  dEtaRed = (input & m_dEtaRedMask) >> m_dPhiRedInWidth;
  etaFine = (input & m_etaFineMask) >> (m_dEtaRedInWidth + m_dPhiRedInWidth);
}
