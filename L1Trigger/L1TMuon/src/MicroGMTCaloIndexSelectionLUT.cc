#include "../interface/MicroGMTCaloIndexSelectionLUT.h"

l1t::MicroGMTCaloIndexSelectionLUT::MicroGMTCaloIndexSelectionLUT (const std::string& fname, int type) : MicroGMTLUT()
{
  if (type == 0) {
    m_angleInWidth = 9;
    m_inputs.push_back(MicroGMTConfiguration::ETA);
  } else {
    m_angleInWidth = 10;
    m_inputs.push_back(MicroGMTConfiguration::PHI);
  }
  
  m_totalInWidth = m_angleInWidth;
  m_outWidth = 6;

  if (fname != std::string("")) {
    load(fname);
  } 
}

int 
l1t::MicroGMTCaloIndexSelectionLUT::lookup(int angle) const 
{
  return lookupPacked(angle);
}
