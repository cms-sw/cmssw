#include "../interface/MicroGMTRankPtQualLUT.h"

l1t::MicroGMTRankPtQualLUT::MicroGMTRankPtQualLUT (const std::string& fname, const unsigned ptFactor, const unsigned qualFactor) : MicroGMTLUT(), m_ptMask(0), m_qualMask(0), m_ptInWidth(9), m_qualInWidth(4), m_ptFactor(ptFactor), m_qualFactor(qualFactor)
{
  m_totalInWidth = m_ptInWidth + m_qualInWidth;
  m_outWidth = 10;

  m_ptMask = (1 << m_ptInWidth) - 1;
  m_qualMask = ((1 << m_qualInWidth) - 1) << m_ptInWidth;
  
  m_inputs.push_back(MicroGMTConfiguration::QUALITY);
  m_inputs.push_back(MicroGMTConfiguration::PT);

  if (fname != std::string("")) {
    load(fname);
  } else {
    initialize();
  }
}

l1t::MicroGMTRankPtQualLUT::MicroGMTRankPtQualLUT (l1t::LUT* lut) : MicroGMTLUT(lut), m_ptMask(0), m_qualMask(0), m_ptInWidth(9), m_qualInWidth(4), m_ptFactor(0), m_qualFactor(0)
{
  m_totalInWidth = m_ptInWidth + m_qualInWidth;
  m_outWidth = 10;

  m_ptMask = (1 << m_ptInWidth) - 1;
  m_qualMask = ((1 << m_qualInWidth) - 1) << m_ptInWidth;

  m_inputs.push_back(MicroGMTConfiguration::QUALITY);
  m_inputs.push_back(MicroGMTConfiguration::PT);

  m_initialized = true;
}

int 
l1t::MicroGMTRankPtQualLUT::lookup(int pt, int qual) const 
{
  // normalize these two to the same scale and then calculate?
  if (m_initialized) {
    return data((unsigned)hashInput(checkedInput(pt, m_ptInWidth), checkedInput(qual, m_qualInWidth)));
  }

  int result = 0;
  result = pt * m_ptFactor + qual * m_qualFactor;
  // normalize to out width
  return result;  
}

int 
l1t::MicroGMTRankPtQualLUT::lookupPacked(int in) const
{
  if (m_initialized) {
    return data((unsigned)in);
  }

  int pt = 0;
  int qual = 0;
  unHashInput(in, pt, qual);
  return lookup(pt, qual);
}

int 
l1t::MicroGMTRankPtQualLUT::hashInput(int pt, int qual) const
{

  int result = 0;
  result += pt;
  result += qual << m_ptInWidth;
  return result;
}

void 
l1t::MicroGMTRankPtQualLUT::unHashInput(int input, int& pt, int& qual) const 
{
  pt = input & m_ptMask;
  qual = (input & m_qualMask) >> m_ptInWidth;
} 
