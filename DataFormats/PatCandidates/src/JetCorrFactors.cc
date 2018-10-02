#include <iomanip>
#include <sstream>
#include <iostream>
#include <algorithm>

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/PatCandidates/interface/JetCorrFactors.h"


using namespace pat;


JetCorrFactors::JetCorrFactors(const std::string& label, const std::vector<CorrectionFactor>& jec): label_(label), jec_(jec)
{
  for(std::vector<CorrectionFactor>::const_iterator corrFactor=jec.begin(); corrFactor!=jec.end(); ++corrFactor){
    if(!isValid(*corrFactor)){
      throw cms::Exception("InvalidRequest") << "You try to create a CorrectionFactor which is neither flavor dependent nor \n"
					     << "flavor independent. The CorrectionFactor should obey the following rules:  \n"
					     << "\n"
					     << " * CorrectionFactor is a std::pair<std::string, std::vector<float> >.      \n"
					     << " * The std::string holds the label of the correction level (following the  \n"
					     << "   conventions of JetMET.                                                  \n"
					     << " * The std::vector<float> holds the correction factors, these factors are  \n"
					     << "   up to the given level. They include all previous correction steps.      \n"
					     << " * The vector has the length *1* for flavor independent correction factors \n"
					     << "   or *5* for flavor dependent correction factors.                         \n"
					     << " * The expected order of flavor dependent correction factors is: NONE,     \n"
					     << "   GLUON, UDS, CHARM, BOTTOM. If follows the JetMET conventions and is     \n"
					     << "   in the Flavor enumerator of the JetCorrFactos class.                    \n"
					     << " * For flavor depdendent correction factors the first entry in the vector  \n"
					     << "   (corresponding to NONE) is invalid and should be set to -1. It will not \n"
					     << "   be considered by the class structure though.                            \n"
					     << "\n"
					     << "Make sure that all elements of the argument vector to this contructor are  \n"
					     << "in accordance with these rules.\n";
    }
  }
}

std::string 
JetCorrFactors::jecFlavor(const Flavor& flavor) const
{
  std::map<Flavor, std::string> flavors;
  flavors[UDS]="uds"; flavors[CHARM]="charm"; flavors[BOTTOM]="bottom"; flavors[GLUON]="gluon"; flavors[NONE]="none"; 
  return flavors.find(flavor)->second;
}

JetCorrFactors::Flavor 
JetCorrFactors::jecFlavor(std::string flavor) const
{
  std::map<std::string, Flavor> flavors;
  std::transform(flavor.begin(), flavor.end(), flavor.begin(), [&](int c){ return std::tolower(c);} );
  flavors["uds"]=UDS; flavors["charm"]=CHARM; flavors["bottom"]=BOTTOM; flavors["gluon"]=GLUON; flavors["none"]=NONE;
  if(flavors.find(flavor)==flavors.end()){
    throw cms::Exception("InvalidRequest") << "You ask for a flavor, which does not exist. Available flavors are: \n"
					   << "'uds', 'charm', 'bottom', 'gluon', 'none', (not case sensitive).   \n";
  }
  return flavors.find(flavor)->second;
}

int  
JetCorrFactors::jecLevel(const std::string& level) const
{
  for(std::vector<CorrectionFactor>::const_iterator corrFactor=jec_.begin(); corrFactor!=jec_.end(); ++corrFactor){
    if(corrFactor->first==level) return (corrFactor-jec_.begin());
  }
  return -1;
}

float 
JetCorrFactors::correction(unsigned int level, Flavor flavor) const 
{
  if(!(level<jec_.size())){
    throw cms::Exception("InvalidRequest") << "You try to call a jet energy correction level wich does not exist. \n"
					   << "Available jet energy correction levels are:                        \n" 
					   << correctionLabelString();    
  }
  if(flavorDependent(jec_.at(level)) && flavor==NONE){
    throw cms::Exception("InvalidRequest") << "You try to call a flavor dependent jet energy correction level:    \n"
					   << "level : " << level << " label: " << jec_.at(level).first << "      \n"
					   << "You need to specify one of the following flavors: GLUON, UDS,      \n"
					   << "CHARM, BOTTOM. \n";
  }
  return flavorDependent(jec_.at(level)) ? jec_.at(level).second.at(flavor) : jec_.at(level).second.at(0);
}

std::string 
JetCorrFactors::correctionLabelString() const 
{
  std::string labels;
  for(std::vector<CorrectionFactor>::const_iterator corrFactor=jec_.begin(); corrFactor!=jec_.end(); ++corrFactor){
    std::stringstream idx; idx << (corrFactor-jec_.begin());
    labels.append(idx.str()).append(" ").append(corrFactor->first).append("\n");
  }
  return labels;
}

std::vector<std::string> 
JetCorrFactors::correctionLabels() const 
{
  std::vector<std::string> labels;
  for(std::vector<CorrectionFactor>::const_iterator corrFactor=jec_.begin(); corrFactor!=jec_.end(); ++corrFactor){
    labels.push_back(corrFactor->first);
  }
  return labels;
}

void
JetCorrFactors::print() const
{
  edm::LogInfo message( "JetCorrFactors" );
  for(std::vector<CorrectionFactor>::const_iterator corrFactor=jec_.begin(); corrFactor!=jec_.end(); ++corrFactor){
    unsigned int corrFactorIdx=corrFactor-jec_.begin();
    message << std::setw(3) << corrFactorIdx << "  " << corrFactor->first;
    if( flavorDependent(*corrFactor) ){
      for(std::vector<float>::const_iterator flavor=corrFactor->second.begin(); flavor!=corrFactor->second.end(); ++flavor){
	unsigned int flavorIdx=flavor-corrFactor->second.begin();
	message << std::setw(10) << correction(corrFactorIdx, (Flavor)flavorIdx);
      }
    }
    else{
      message << std::setw(10) << correction (corrFactor-jec_.begin(), NONE); 
    }
    message << "\n";
  }
}
