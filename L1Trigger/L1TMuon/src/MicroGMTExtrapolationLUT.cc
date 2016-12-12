#include "../interface/MicroGMTExtrapolationLUT.h"

l1t::MicroGMTExtrapolationLUT::MicroGMTExtrapolationLUT (const std::string& fname, const int type) : MicroGMTLUT(), m_etaRedInWidth(6), m_ptRedInWidth(6)
{
  m_totalInWidth = m_ptRedInWidth + m_etaRedInWidth;
  if (type == MicroGMTConfiguration::ETA_OUT) {
    m_outWidth = 4;
  } else {
    m_outWidth = 3;
  }

  m_ptRedMask = (1 << m_ptRedInWidth) - 1;
  m_etaRedMask = ((1 << m_etaRedInWidth) - 1) << m_ptRedInWidth;
  
  m_inputs.push_back(MicroGMTConfiguration::ETA_COARSE);
  m_inputs.push_back(MicroGMTConfiguration::PT);

  if (fname != std::string("")) {
    load(fname);
  } 
}

l1t::MicroGMTExtrapolationLUT::MicroGMTExtrapolationLUT (l1t::LUT* lut, const int type) : MicroGMTLUT(lut), m_etaRedInWidth(6), m_ptRedInWidth(6)
{
  m_totalInWidth = m_ptRedInWidth + m_etaRedInWidth;
  if (type == MicroGMTConfiguration::ETA_OUT) {
    m_outWidth = 4;
  } else {
    m_outWidth = 3;
  }

  m_ptRedMask = (1 << m_ptRedInWidth) - 1;
  m_etaRedMask = ((1 << m_etaRedInWidth) - 1) << m_ptRedInWidth;

  m_inputs.push_back(MicroGMTConfiguration::ETA_COARSE);
  m_inputs.push_back(MicroGMTConfiguration::PT);

  m_initialized = true;
}

int 
l1t::MicroGMTExtrapolationLUT::lookup(int eta, int pt) const 
{
  // normalize these two to the same scale and then calculate?
  if (m_initialized) {
    // unsigned eta_twocomp = MicroGMTConfiguration::getTwosComp(eta, m_etaRedInWidth);
    return lookupPacked(hashInput(checkedInput(eta, m_etaRedInWidth), checkedInput(pt, m_ptRedInWidth)));
  }
  int result = 0;
  // normalize to out width
  return result;
}

int 
l1t::MicroGMTExtrapolationLUT::hashInput(int eta, int pt) const
{
  int result = 0;
  result += eta << m_ptRedInWidth;
  result += pt;
  return result;
}

void 
l1t::MicroGMTExtrapolationLUT::unHashInput(int input, int& eta, int& pt) const 
{
  pt = input & m_ptRedMask;
  eta = (input & m_etaRedMask) >> m_ptRedInWidth;
} 
