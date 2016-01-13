#include "../interface/MicroGMTAbsoluteIsolationCheckLUT.h"

l1t::MicroGMTAbsoluteIsolationCheckLUT::MicroGMTAbsoluteIsolationCheckLUT(const std::string& fname) : m_energySumInWidth(5) 
{
  m_totalInWidth = m_energySumInWidth;

  if (fname != std::string("")) {
    load(fname);
  } 

  m_inputs.push_back(MicroGMTConfiguration::PT);
  m_inputs.push_back(MicroGMTConfiguration::ETA);
}

int 
l1t::MicroGMTAbsoluteIsolationCheckLUT::lookup(int energySum) const 
{
  return lookupPacked(checkedInput(energySum, m_energySumInWidth));
}
