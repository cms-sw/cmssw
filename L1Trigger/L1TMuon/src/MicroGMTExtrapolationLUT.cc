#include "../interface/MicroGMTExtrapolationLUT.h"

l1t::MicroGMTExtrapolationLUT::MicroGMTExtrapolationLUT (const std::string& fname) : m_etaRedInWidth(6), m_ptRedInWidth(6)
{
  m_totalInWidth = m_ptRedInWidth + m_etaRedInWidth;

  m_ptRedMask = (1 << m_ptRedInWidth) - 1;
  m_etaRedMask = ((1 << m_etaRedInWidth) - 1) << m_ptRedInWidth;
  
  if (fname != std::string("")) {
    load(fname);
  } 
  m_inputs.push_back(MicroGMTConfiguration::PT);
  m_inputs.push_back(MicroGMTConfiguration::ETA);
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
  eta = input & m_etaRedMask;
  pt = input >> m_etaRedInWidth;
} 
