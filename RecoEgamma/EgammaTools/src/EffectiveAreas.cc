#include "RecoEgamma/EgammaTools/interface/EffectiveAreas.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <cmath>
#include <fstream>
#include <string>
#include <sstream>

EffectiveAreas::EffectiveAreas(const std::string& filename):
  filename_(filename)
{

  // Open the file with the effective area constants
  std::ifstream inputFile;
  inputFile.open(filename_.c_str());
  if( !inputFile.is_open() )
    throw cms::Exception("EffectiveAreas config failure")
      << "failed to open the file " << filename_ << std::endl;
  
  // Read file line by line
  std::string line;
  const float undef = -999;
  while( getline(inputFile, line) ){
    if(line[0]=='#') continue; // skip the comments lines
    float etaMin = undef, etaMax = undef, effArea = undef;
    std::stringstream ss(line);
    ss >>  etaMin >> etaMax >> effArea;
    // In case if the format is messed up, there are letters
    // instead of numbers, or not exactly three numbers in the line,
    // it is likely that one or more of these vars never changed
    // the original "undef" value:
    if( etaMin==undef || etaMax==undef || effArea==undef )
      throw cms::Exception("EffectiveAreas config failure")
	<< "wrong file format, file name " << filename_ << std::endl;
    
    absEtaMin_          .push_back( etaMin );
    absEtaMax_          .push_back( etaMax );
    effectiveAreaValues_.push_back( effArea );
  }

  // Extra consistency checks are in the function below.
  // If any of them fail, an exception is thrown.
  checkConsistency();
}

EffectiveAreas::~EffectiveAreas(){

  absEtaMin_.clear();
  absEtaMax_.clear();
  effectiveAreaValues_.clear();

}

// Return effective area for given eta
const float EffectiveAreas::getEffectiveArea(float eta) const{

  float effArea = 0;
  uint nEtaBins = absEtaMin_.size();
  for(uint iEta = 0; iEta<nEtaBins; iEta++){
    if( std::abs(eta) >= absEtaMin_[iEta]
	&& std::abs(eta) < absEtaMax_[iEta] ){
      effArea = effectiveAreaValues_[iEta];
      break;
    }
  }

  return effArea;
}

void EffectiveAreas::printEffectiveAreas() const {

  printf("EffectiveAreas: source file %s\n", filename_.c_str());
  printf("  eta_min   eta_max    effective area\n");
  uint nEtaBins = absEtaMin_.size();
  for(uint iEta = 0; iEta<nEtaBins; iEta++){
    printf("  %8.4f    %8.4f   %8.5f\n",
	   absEtaMin_[iEta], absEtaMax_[iEta],
	   effectiveAreaValues_[iEta]);
  }

}

// Basic common sense checks
void EffectiveAreas::checkConsistency() const {

  // There should be at least one eta range with one constant
  if( effectiveAreaValues_.size() == 0 )
    throw cms::Exception("EffectiveAreas config failure")
      << "found no effective area constans in the file " 
      << filename_ << std::endl;

  uint nEtaBins = absEtaMin_.size();
  for(uint iEta = 0; iEta<nEtaBins; iEta++){

    // The low limit should be lower than the upper limit
    if( !( absEtaMin_[iEta] < absEtaMax_[iEta] ) )
      throw cms::Exception("EffectiveAreas config failure")
	<< "eta ranges improperly defined (min>max) in the file" 
	<< filename_ << std::endl;

    // The low limit of the next range should be (near) equal to the
    // upper limit of the previous range
    if( iEta != nEtaBins-1 ) // don't do the check for the last bin
      if( !( absEtaMin_[iEta+1] - absEtaMax_[iEta] < 0.0001 ) )
	throw cms::Exception("EffectiveAreas config failure")
	  << "eta ranges improperly defined (disjointed) in the file " 
	  << filename_ << std::endl;

    // The effective area should be non-negative number,
    // and should be less than the whole calorimeter area 
    // eta range -2.5 to 2.5, phi 0 to 2pi => Amax = 5*2*pi ~= 31.4
    if( !( effectiveAreaValues_[iEta] >= 0
	   && effectiveAreaValues_[iEta] < 31.4 ) )
      throw cms::Exception("EffectiveAreas config failure")
	<< "effective area values are too large or negative in the file"
	<< filename_ << std::endl;
  }

}
