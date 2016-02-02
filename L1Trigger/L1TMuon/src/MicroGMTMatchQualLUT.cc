#include "../interface/MicroGMTMatchQualLUT.h"
#include "TMath.h"

l1t::MicroGMTMatchQualLUT::MicroGMTMatchQualLUT (const std::string& fname, cancel_t cancelType) :
  m_dEtaRedMask(0), m_dPhiRedMask(0), m_dEtaRedInWidth(4), m_dPhiRedInWidth(3), m_etaScale(0), m_phiScale(0), m_cancelType(cancelType)
{
  m_totalInWidth = m_dPhiRedInWidth + m_dEtaRedInWidth;

  m_dEtaRedMask = (1 << m_dEtaRedInWidth) - 1;
  m_dPhiRedMask = (1 << (m_totalInWidth - 1)) - m_dEtaRedMask - 1;

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

l1t::MicroGMTMatchQualLUT::~MicroGMTMatchQualLUT ()
{

}


int
l1t::MicroGMTMatchQualLUT::lookup(int dEtaRed, int dPhiRed) const
{
  // normalize these two to the same scale and then calculate?
  if (m_initialized) {
    return data((unsigned)hashInput(checkedInput(dEtaRed, m_dEtaRedInWidth), checkedInput(dPhiRed, m_dPhiRedInWidth)));
  }
  double dEta = dEtaRed*m_etaScale;
  double dPhi = dPhiRed*m_phiScale;

  double dR = std::sqrt(dEta*dEta + dPhi*dPhi);

  int retVal = dR < 0.1 ? 1 : 0;
  // should we need customisation for the different track finder cancellations:
  // switch (m_cancelType) {
  //   case bmtf_bmtf:
  //     retVal = dR < 0.1 ? 1 : 0;
  //   case default:
  //     retVal = dR < 0.1 ? 1 : 0
  // }

  return retVal;
}
int
l1t::MicroGMTMatchQualLUT::lookupPacked(int in) const {
  if (m_initialized) {
    return data((unsigned)in);
  }

  int dEtaRed = 0;
  int dPhiRed = 0;
  unHashInput(in, dEtaRed, dPhiRed);
  return lookup(dEtaRed, dPhiRed);
}

int
l1t::MicroGMTMatchQualLUT::hashInput(int dEtaRed, int dPhiRed) const
{

  int result = 0;
  result += dEtaRed;
  result += dPhiRed << m_dEtaRedInWidth;
  return result;
}

void
l1t::MicroGMTMatchQualLUT::unHashInput(int input, int& dEtaRed, int& dPhiRed) const
{
  dEtaRed = input & m_dEtaRedMask;
  dPhiRed = (input & m_dPhiRedMask) >> m_dEtaRedInWidth;
}
