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
  GEMValidator* validator = new GEMValidator();
//   validator->produceSimHitValidationPlots(GEMValidator::Muon); 
//   validator->produceSimHitValidationPlots(GEMValidator::NonMuon); 
//   validator->produceSimHitValidationPlots(GEMValidator::All); 
//   validator->produceDigiValidationPlots(); 
//   validator->produceGEMCSCPadDigiValidationPlots("GEMCSCPadDigiTree");  
//   validator->produceGEMCSCPadDigiValidationPlots("GEMCSCCoPadDigiTree");  
  validator->produceTrackValidationPlots(); 
  return 0;
}
