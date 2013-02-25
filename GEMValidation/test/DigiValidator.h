#ifndef _DigiValidator_h_
#define _DigiValidator_h_

#include "BaseValidator.h"

class DigiValidator : public BaseValidator
{
 public:
  DigiValidator();
  ~DigiValidator();

  void makeValidationPlots();
  void makeGEMCSCPadDigiValidationPlots(const std::string treeName);
  void makeTrackValidationPlots();
  void makeValidationReport();
  template<typename T> const std::string to_string( T const& value );
 private:
  
};

#endif
