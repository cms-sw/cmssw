#include "DataFormats/SiPixelDigi/interface/SiPixelCalibDigiError.h"


std::string SiPixelCalibDigiError::printError() const
{
  std::string result = "unknown error";
  switch (fErrorType) {
  case(1):{
    result="arrived at unexpected pattern.";
    break;
  }
  case(2):{
    result="pixel is not in pattern at all.";
    break;
  }
  }

  return result;
}
