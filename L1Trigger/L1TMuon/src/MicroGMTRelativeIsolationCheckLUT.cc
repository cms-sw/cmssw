#include "../interface/MicroGMTRelativeIsolationCheckLUT.h"

l1t::MicroGMTRelativeIsolationCheckLUT::MicroGMTRelativeIsolationCheckLUT(const std::string& fname) : m_energySumInWidth(5), m_ptInWidth(9)
{
  m_totalInWidth = m_ptInWidth + m_energySumInWidth;

  m_ptMask = (1 << m_ptInWidth) - 1;
  m_energySumMask = (1 << (m_totalInWidth - 1)) - m_ptMask;
  if (fname != std::string("")) {
    load(fname);
  } 
  m_inputs.push_back(MicroGMTConfiguration::PT);
  m_inputs.push_back(MicroGMTConfiguration::ETA);
}

int 
l1t::MicroGMTRelativeIsolationCheckLUT::lookup(int energySum, int pt) const 
{
  // normalize these two to the same scale and then calculate?
  return lookupPacked(hashInput(checkedInput(energySum, m_energySumInWidth), checkedInput(pt, m_ptInWidth)));
}

int 
l1t::MicroGMTRelativeIsolationCheckLUT::hashInput(int energySum, int pT) const
{
  int result = 0;
  result += energySum << m_ptInWidth;
  result += pT;
  return result;
}

void 
l1t::MicroGMTRelativeIsolationCheckLUT::unHashInput(int input, int& energySum, int& pt) const 
{
  energySum = input & m_energySumMask;
  pt = (input & m_ptMask) >> m_energySumInWidth;
} 
