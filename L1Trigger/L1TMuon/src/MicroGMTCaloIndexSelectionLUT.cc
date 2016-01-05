#include "../interface/MicroGMTCaloIndexSelectionLUT.h"

l1t::MicroGMTCaloIndexSelectionLUT::MicroGMTCaloIndexSelectionLUT (const std::string& fname, int type) {
  if (type == 0) {
    m_angleInWidth = 9;
  } else {
    m_angleInWidth = 10;
  }
  
  m_totalInWidth = m_angleInWidth;
  if (fname != std::string("")) {
    load(fname);
  } 

  m_inputs.push_back(MicroGMTConfiguration::PT);
  m_inputs.push_back(MicroGMTConfiguration::ETA);
}

int 
l1t::MicroGMTCaloIndexSelectionLUT::lookup(int angle) const 
{
  return lookupPacked(angle);
}
