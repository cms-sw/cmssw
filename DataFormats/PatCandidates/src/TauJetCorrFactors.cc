#include "DataFormats/PatCandidates/interface/TauJetCorrFactors.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <iomanip>
#include <iostream>
#include <sstream>

using namespace pat;

TauJetCorrFactors::TauJetCorrFactors(const std::string& label, const std::vector<CorrectionFactor>& jec)
  : label_(label), 
    jec_(jec)
{}

int  
TauJetCorrFactors::jecLevel(const std::string& level) const
{
  for ( std::vector<CorrectionFactor>::const_iterator corrFactor = jec_.begin(); 
	corrFactor != jec_.end(); ++corrFactor ) {
    if ( corrFactor->first == level ) return (corrFactor-jec_.begin());
  }
  return -1;
}

float 
TauJetCorrFactors::correction(unsigned int level) const 
{
  if ( !(level < jec_.size()) ) {
    throw cms::Exception("InvalidRequest") 
      << "You try to call a jet energy correction level wich does not exist. \n"
      << "Available jet energy correction levels are:                        \n" 
      << correctionLabelString();    
  }
  return jec_.at(level).second;
}

std::string 
TauJetCorrFactors::correctionLabelString() const 
{
  std::string labels;
  for ( std::vector<CorrectionFactor>::const_iterator corrFactor = jec_.begin(); 
	corrFactor != jec_.end(); ++corrFactor ) {
    std::stringstream idx; idx << (corrFactor-jec_.begin());
    labels.append(idx.str()).append(" ").append(corrFactor->first).append("\n");
  }
  return labels;
}

std::vector<std::string> 
TauJetCorrFactors::correctionLabels() const 
{
  std::vector<std::string> labels;
  for ( std::vector<CorrectionFactor>::const_iterator corrFactor = jec_.begin(); 
	corrFactor != jec_.end(); ++corrFactor ) {
    labels.push_back(corrFactor->first);
  }
  return labels;
}

void
TauJetCorrFactors::print() const
{
  edm::LogInfo message("JetCorrFactors");
  for ( std::vector<CorrectionFactor>::const_iterator corrFactor = jec_.begin(); 
	corrFactor != jec_.end(); ++corrFactor ) {
    unsigned int corrFactorIdx = corrFactor-jec_.begin();
    message << std::setw(3) << corrFactorIdx << "  " << corrFactor->first;
    message << std::setw(10) << correction (corrFactor-jec_.begin()); 
    message << "\n";
  }
}
