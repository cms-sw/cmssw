#include "../interface/MicroGMTCaloIndexSelectionLUT.h"

l1t::MicroGMTCaloIndexSelectionLUT::MicroGMTCaloIndexSelectionLUT (const std::string& fname, int type) : MicroGMTLUT()
{
  if (type == MicroGMTConfiguration::ETA) {
    m_angleInWidth = 9;
    m_outWidth = 5;
    m_inputs.push_back(MicroGMTConfiguration::ETA);
  } else {
    m_angleInWidth = 10;
    m_outWidth = 6;
    m_inputs.push_back(MicroGMTConfiguration::PHI);
  }

  m_totalInWidth = m_angleInWidth;

  if (fname != std::string("")) {
    load(fname);
  } 
}

l1t::MicroGMTCaloIndexSelectionLUT::MicroGMTCaloIndexSelectionLUT (l1t::LUT* lut, int type) : MicroGMTLUT(lut)
{
  if (type == MicroGMTConfiguration::ETA) {
    m_angleInWidth = 9;
    m_outWidth = 5;
    m_inputs.push_back(MicroGMTConfiguration::ETA);
  } else {
    m_angleInWidth = 10;
    m_outWidth = 6;
    m_inputs.push_back(MicroGMTConfiguration::PHI);
  }

  m_totalInWidth = m_angleInWidth;

  m_initialized = true;
}

int 
l1t::MicroGMTCaloIndexSelectionLUT::lookup(int angle) const 
{
  return lookupPacked(angle);
}
