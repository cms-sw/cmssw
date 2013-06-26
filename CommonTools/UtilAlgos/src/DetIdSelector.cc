#include "CommonTools/UtilAlgos/interface/DetIdSelector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


DetIdSelector::DetIdSelector():
  m_selections(),m_masks()
{}

DetIdSelector::DetIdSelector(const std::string& selstring):
  m_selections(),m_masks()
{
  addSelection(selstring);
}

DetIdSelector::DetIdSelector(const std::vector<std::string>& selstrings):
  m_selections(),m_masks()
{
  addSelection(selstrings);
}

DetIdSelector::DetIdSelector(const edm::ParameterSet& selconfig):
  m_selections(),m_masks()
{

  const std::vector<std::string> selstrings = selconfig.getUntrackedParameter<std::vector<std::string> >("selection");
  addSelection(selstrings);

}

void DetIdSelector::addSelection(const std::string& selstring) {

  unsigned int selection;
  unsigned int mask;

  if(selstring.substr(0,2) == "0x") {
    sscanf(selstring.c_str(),"%x-%x",&mask,&selection);
  }
  else {
    sscanf(selstring.c_str(),"%u-%u",&mask,&selection);
  }

  m_selections.push_back(selection);
  m_masks.push_back(mask);

  LogDebug("Selection added") << "Selection " << selection << " with mask " << mask << " added";

}

void DetIdSelector::addSelection(const std::vector<std::string>& selstrings) {

  for(std::vector<std::string>::const_iterator selstring=selstrings.begin();selstring!=selstrings.end();++selstring) {
    addSelection(*selstring);
  }

}

bool DetIdSelector::isSelected(const unsigned int& rawid) const {

  for(unsigned int i=0; i<m_selections.size() ; ++i) {
    if((m_masks[i] & rawid) == m_selections[i]) return true;
  }

  return false;
}

bool DetIdSelector::isSelected(const DetId& detid) const {

  return isSelected(detid.rawId());

}

bool DetIdSelector::operator()(const DetId& detid) const {

  return isSelected(detid.rawId());

}

bool DetIdSelector::operator()(const unsigned int& rawid) const {

  return isSelected(rawid);

}



