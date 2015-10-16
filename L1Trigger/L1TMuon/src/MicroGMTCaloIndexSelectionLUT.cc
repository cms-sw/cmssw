#include "../interface/MicroGMTCaloIndexSelectionLUT.h"

l1t::MicroGMTCaloIndexSelectionLUT::MicroGMTCaloIndexSelectionLUT (const edm::ParameterSet& iConfig, const std::string& setName, int type) {
  getParameters(iConfig, setName.c_str(), type);
}

l1t::MicroGMTCaloIndexSelectionLUT::MicroGMTCaloIndexSelectionLUT (const edm::ParameterSet& iConfig, const char* setName, int type) {
  getParameters(iConfig, setName, type);

}

void 
l1t::MicroGMTCaloIndexSelectionLUT::getParameters (const edm::ParameterSet& iConfig, const char* setName, int type) {
  edm::ParameterSet config = iConfig.getParameter<edm::ParameterSet>(setName);
  if (type == 0) {
    m_angleInWidth = config.getParameter<int>("eta_in_width");
  } else {
    m_angleInWidth = config.getParameter<int>("phi_in_width");
  }
  
  m_totalInWidth = m_angleInWidth;
  std::string m_fname = config.getParameter<std::string>("filename");
  if (m_fname != std::string("")) {
    load(m_fname);
  } 

  m_inputs.push_back(MicroGMTConfiguration::PT);
  m_inputs.push_back(MicroGMTConfiguration::ETA);
}


l1t::MicroGMTCaloIndexSelectionLUT::~MicroGMTCaloIndexSelectionLUT ()
{

}


int 
l1t::MicroGMTCaloIndexSelectionLUT::lookup(int angle) const 
{
  return lookupPacked(angle);
}
