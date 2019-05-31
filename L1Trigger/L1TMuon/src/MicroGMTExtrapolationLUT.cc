#include "L1Trigger/L1TMuon/interface/MicroGMTExtrapolationLUT.h"

l1t::MicroGMTExtrapolationLUT::MicroGMTExtrapolationLUT(const std::string& fname,
                                                        const int outWidth,
                                                        const int etaRedInWidth,
                                                        const int ptRedInWidth)
    : MicroGMTLUT(), m_etaRedInWidth(etaRedInWidth), m_ptRedInWidth(ptRedInWidth) {
  m_totalInWidth = m_ptRedInWidth + m_etaRedInWidth;
  m_outWidth = outWidth;

  m_ptRedMask = (1 << m_ptRedInWidth) - 1;
  m_etaRedMask = ((1 << m_etaRedInWidth) - 1) << m_ptRedInWidth;

  m_inputs.push_back(MicroGMTConfiguration::ETA_COARSE);
  m_inputs.push_back(MicroGMTConfiguration::PT);

  if (fname != std::string("")) {
    load(fname);
  }
}

l1t::MicroGMTExtrapolationLUT::MicroGMTExtrapolationLUT(l1t::LUT* lut,
                                                        const int outWidth,
                                                        const int etaRedInWidth,
                                                        const int ptRedInWidth)
    : MicroGMTLUT(lut), m_etaRedInWidth(etaRedInWidth), m_ptRedInWidth(ptRedInWidth) {
  m_totalInWidth = m_ptRedInWidth + m_etaRedInWidth;
  m_outWidth = outWidth;

  m_ptRedMask = (1 << m_ptRedInWidth) - 1;
  m_etaRedMask = ((1 << m_etaRedInWidth) - 1) << m_ptRedInWidth;

  m_inputs.push_back(MicroGMTConfiguration::ETA_COARSE);
  m_inputs.push_back(MicroGMTConfiguration::PT);

  m_initialized = true;
}

int l1t::MicroGMTExtrapolationLUT::lookup(int eta, int pt) const {
  // normalize these two to the same scale and then calculate?
  if (m_initialized) {
    // unsigned eta_twocomp = MicroGMTConfiguration::getTwosComp(eta, m_etaRedInWidth);
    return lookupPacked(hashInput(checkedInput(eta, m_etaRedInWidth), checkedInput(pt, m_ptRedInWidth)));
  }
  int result = 0;
  // normalize to out width
  return result;
}

int l1t::MicroGMTExtrapolationLUT::hashInput(int eta, int pt) const {
  int result = 0;
  result += eta << m_ptRedInWidth;
  result += pt;
  return result;
}

void l1t::MicroGMTExtrapolationLUT::unHashInput(int input, int& eta, int& pt) const {
  pt = input & m_ptRedMask;
  eta = (input & m_etaRedMask) >> m_ptRedInWidth;
}

int l1t::MicroGMTExtrapolationLUT::getEtaRedInWidth() const { return m_etaRedInWidth; }

int l1t::MicroGMTExtrapolationLUT::getPtRedInWidth() const { return m_ptRedInWidth; }
