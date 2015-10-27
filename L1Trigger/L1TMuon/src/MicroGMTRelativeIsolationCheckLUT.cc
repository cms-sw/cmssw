#include "../interface/MicroGMTRelativeIsolationCheckLUT.h"

l1t::MicroGMTRelativeIsolationCheckLUT::MicroGMTRelativeIsolationCheckLUT (const edm::ParameterSet& iConfig, const std::string& setName) 
{
  getParameters(iConfig, setName.c_str());
}

l1t::MicroGMTRelativeIsolationCheckLUT::MicroGMTRelativeIsolationCheckLUT (const edm::ParameterSet& iConfig, const char* setName) 
{
  getParameters(iConfig, setName);
}

void 
l1t::MicroGMTRelativeIsolationCheckLUT::getParameters (const edm::ParameterSet& iConfig, const char* setName) 
{
  edm::ParameterSet config = iConfig.getParameter<edm::ParameterSet>(setName);
  m_energySumInWidth = config.getParameter<int>("areaSum_in_width");
  m_ptInWidth = config.getParameter<int>("pT_in_width");
  
  m_totalInWidth = m_ptInWidth + m_energySumInWidth;

  m_ptMask = (1 << m_ptInWidth) - 1;
  m_energySumMask = (1 << (m_totalInWidth - 1)) - m_ptMask;
  std::string m_fname = config.getParameter<std::string>("filename");
  if (m_fname != std::string("")) {
    load(m_fname);
  } 
  m_inputs.push_back(MicroGMTConfiguration::PT);
  m_inputs.push_back(MicroGMTConfiguration::ETA);
}


l1t::MicroGMTRelativeIsolationCheckLUT::~MicroGMTRelativeIsolationCheckLUT ()
{

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