#include "../interface/MicroGMTRankPtQualLUT.h"

l1t::MicroGMTRankPtQualLUT::MicroGMTRankPtQualLUT (const edm::ParameterSet& iConfig) {
  edm::ParameterSet config = iConfig.getParameter<edm::ParameterSet>("SortRankLUTSettings");
  m_ptInWidth = config.getParameter<int>("pT_in_width");
  m_qualInWidth = config.getParameter<int>("qual_in_width");
  
  m_totalInWidth = m_ptInWidth + m_qualInWidth;

  m_ptMask = (1 << m_ptInWidth) - 1;
  m_qualMask = (1 << (m_totalInWidth - 1)) - m_ptMask - 1;
  
  m_inputs.push_back(MicroGMTConfiguration::PT);
  std::string m_fname = config.getParameter<std::string>("filename");
  if (m_fname != std::string("")) {
    load(m_fname);
  } else {
    initialize();
  }
}

l1t::MicroGMTRankPtQualLUT::MicroGMTRankPtQualLUT () : MicroGMTLUT(), m_ptMask(0), m_qualMask(0), m_ptInWidth(9), m_qualInWidth(4)
{
  m_ptMask = (1 << m_ptInWidth) - 1;
  m_qualMask = (1 << (m_totalInWidth - 1)) - m_ptMask - 1;
  m_outWidth = 10;
  m_totalInWidth = m_ptInWidth + m_qualInWidth;
} 

l1t::MicroGMTRankPtQualLUT::~MicroGMTRankPtQualLUT ()
{

}

int 
l1t::MicroGMTRankPtQualLUT::lookup(int pt, int qual) const 
{
  // normalize these two to the same scale and then calculate?
  if (m_initialized) {
    return m_contents.at(hashInput(checkedInput(pt, m_ptInWidth), checkedInput(qual, m_qualInWidth)));
  }

  int result = 0;
  result = pt + (qual << 2); 
  // normalize to out width
  return result;  
}

int 
l1t::MicroGMTRankPtQualLUT::lookupPacked(int in) const {
  if (m_initialized) {
    return m_contents.at(in);
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