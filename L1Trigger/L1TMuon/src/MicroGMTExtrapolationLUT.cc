#include "../interface/MicroGMTExtrapolationLUT.h"

l1t::MicroGMTExtrapolationLUT::MicroGMTExtrapolationLUT (const edm::ParameterSet& iConfig, const std::string& setName, const int type) {
  getParameters(iConfig, setName.c_str(), type);
}

l1t::MicroGMTExtrapolationLUT::MicroGMTExtrapolationLUT (const edm::ParameterSet& iConfig, const char* setName, const int type) {
  getParameters(iConfig, setName, type);
}

void 
l1t::MicroGMTExtrapolationLUT::getParameters (const edm::ParameterSet& iConfig, const char* setName, const int type) {
  edm::ParameterSet config = iConfig.getParameter<edm::ParameterSet>(setName);
  
  m_etaRedInWidth = config.getParameter<int>("etaAbsRed_in_width");
  m_ptRedInWidth = config.getParameter<int>("pTred_in_width");
  
  m_totalInWidth = m_ptRedInWidth + m_etaRedInWidth;

  m_ptRedMask = (1 << m_ptRedInWidth) - 1;
  m_etaRedMask = ((1 << m_etaRedInWidth) - 1) << m_ptRedInWidth;
  
  std::string m_fname = config.getParameter<std::string>("filename");
  if (m_fname != std::string("")) {
    load(m_fname);
  } 
  m_inputs.push_back(MicroGMTConfiguration::PT);
  m_inputs.push_back(MicroGMTConfiguration::ETA);
}


l1t::MicroGMTExtrapolationLUT::~MicroGMTExtrapolationLUT ()
{

}


int 
l1t::MicroGMTExtrapolationLUT::lookup(int eta, int pt) const 
{
  // normalize these two to the same scale and then calculate?
  if (m_initialized) {
    // unsigned eta_twocomp = MicroGMTConfiguration::getTwosComp(eta, m_etaRedInWidth);
    return lookupPacked(hashInput(checkedInput(eta, m_etaRedInWidth), checkedInput(pt, m_ptRedInWidth)));
  }
  int result = 0;
  // normalize to out width
  return result;
}

int 
l1t::MicroGMTExtrapolationLUT::hashInput(int eta, int pt) const
{
  int result = 0;
  result += eta << m_ptRedInWidth;
  result += pt;
  return result;
}

void 
l1t::MicroGMTExtrapolationLUT::unHashInput(int input, int& eta, int& pt) const 
{
  eta = input & m_etaRedMask;
  pt = input >> m_etaRedInWidth;
} 