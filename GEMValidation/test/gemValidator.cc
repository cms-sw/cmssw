
#include "GEMValidator.h"
#include <iostream>

template < typename T > std::string to_string( T const & value )
{
  std::stringstream sstr;
  sstr << value;
  return sstr.str();
}

int main(int argc, char* argv[])
{
  if(argc!=2){
    std::cout << "Usage: ./gemValidator Identifier Selection" << std::endl
	      << "Identifier: identifier of the SimHit and Digi ROOT file" << std::endl
 	      << "SimHit type: {MUON = 0, NON_MUON = 1, ALL = 2}" << std::endl;
    exit(1);
  }
  
  const std::string identifier(to_string(argv[1]));
  GEMValidator* validator = new GEMValidator(identifier);
  validator->produceSimHitValidationPlots(GEMValidator::Muon); 
  validator->produceSimHitValidationPlots(GEMValidator::NonMuon); 
  validator->produceSimHitValidationPlots(GEMValidator::All); 
  validator->produceDigiValidationPlots(); 
  validator->produceTrackValidationPlots(); 
  return 0;
}

