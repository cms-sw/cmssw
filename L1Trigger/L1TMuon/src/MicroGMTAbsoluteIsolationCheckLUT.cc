#include "../interface/MicroGMTAbsoluteIsolationCheckLUT.h"

l1t::MicroGMTAbsoluteIsolationCheckLUT::MicroGMTAbsoluteIsolationCheckLUT(const std::string& fname) : MicroGMTLUT(), m_energySumInWidth(5)
{
  m_totalInWidth = m_energySumInWidth;
  m_outWidth = 1;

  if (fname != std::string("")) {
    load(fname);
  } 

  m_inputs.push_back(MicroGMTConfiguration::ENERGYSUM);
}

l1t::MicroGMTAbsoluteIsolationCheckLUT::MicroGMTAbsoluteIsolationCheckLUT(l1t::LUT* lut) : MicroGMTLUT(lut), m_energySumInWidth(5)
{
  m_totalInWidth = m_energySumInWidth;
  m_outWidth = 1;

  m_inputs.push_back(MicroGMTConfiguration::ENERGYSUM);
  m_initialized = true;
}

int 
l1t::MicroGMTAbsoluteIsolationCheckLUT::lookup(int energySum) const 
{
  return lookupPacked(checkedInput(energySum, m_energySumInWidth));
}
