#ifndef ErrorChecker_H
#define ErrorChecker_H
/** \class ErrorChecker
 *
 *  
 */

#include "DataFormats/SiPixelRawData/interface/SiPixelRawDataError.h"

#include <boost/cstdint.hpp>
#include <vector>
#include <map>

class FEDRawData;

class SiPixelFrameConverter;

class ErrorChecker {

public:
  typedef unsigned int Word32;
  typedef long long Word64;
  typedef std::vector<SiPixelRawDataError> DetErrors;
  typedef std::map<uint32_t, DetErrors> Errors;

  ErrorChecker();

  void setErrorStatus(bool ErrorStatus);

  bool checkHeader(int fedId, const Word64* header, Errors& errors);

  bool checkTrailer(int fedId, int nWords, const Word64* trailer, Errors& errors);

  bool checkROC(int fedId, const SiPixelFrameConverter* converter, 
		Word32 errorWord, Errors& errors);

  void conversionError(int fedId, const SiPixelFrameConverter* converter, 
		       int status, Word32 errorWord, Errors& errors);

private:
  bool includeErrors;

  uint32_t errorDetId(const SiPixelFrameConverter* converter, 
		      int errorType, const Word32 & word) const;

  static const int LINK_bits,  ROC_bits,  DCOL_bits,  PXID_bits,  ADC_bits;
  static const int LINK_shift, ROC_shift, DCOL_shift, PXID_shift, ADC_shift;
  static const uint32_t dummyDetId;
};

#endif
