#include "../interface/MicroGMTAbsoluteIsolationCheckLUT.h"

l1t::MicroGMTAbsoluteIsolationCheckLUT::MicroGMTAbsoluteIsolationCheckLUT (const edm::ParameterSet& iConfig, const std::string& setName) 
{
  getParameters(iConfig, setName.c_str());
}

l1t::MicroGMTAbsoluteIsolationCheckLUT::MicroGMTAbsoluteIsolationCheckLUT (const edm::ParameterSet& iConfig, const char* setName) 
{
  getParameters(iConfig, setName);
}

void 
l1t::MicroGMTAbsoluteIsolationCheckLUT::getParameters (const edm::ParameterSet& iConfig, const char* setName) 
{
  edm::ParameterSet config = iConfig.getParameter<edm::ParameterSet>(setName);
  m_energySumInWidth = config.getParameter<int>("areaSum_in_width");
  
  m_totalInWidth = m_energySumInWidth;

  std::string m_fname = config.getParameter<std::string>("filename");
  if (m_fname != std::string("")) {
    load(m_fname);
  } 

  m_inputs.push_back(MicroGMTConfiguration::PT);
  m_inputs.push_back(MicroGMTConfiguration::ETA);
}


l1t::MicroGMTAbsoluteIsolationCheckLUT::~MicroGMTAbsoluteIsolationCheckLUT ()
{

}


int 
l1t::MicroGMTAbsoluteIsolationCheckLUT::lookup(int energySum) const 
{
  return lookupPacked(checkedInput(energySum, m_energySumInWidth));
}
