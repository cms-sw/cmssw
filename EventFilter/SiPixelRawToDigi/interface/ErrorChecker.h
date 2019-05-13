#ifndef ErrorChecker_H
#define ErrorChecker_H
/** \class ErrorChecker
 *
 *  
 */

#include "EventFilter/SiPixelRawToDigi/interface/ErrorCheckerBase.h"
#include "FWCore/Utilities/interface/typedefs.h"


class ErrorChecker : public ErrorCheckerBase {

public:
  typedef cms_uint32_t Word32;
  typedef cms_uint64_t Word64;

  typedef std::vector<SiPixelRawDataError> DetErrors;
  typedef std::map<cms_uint32_t, DetErrors> Errors;

  ErrorChecker();

  void setErrorStatus(bool ErrorStatus) override;

  bool checkCRC(bool& errorsInEvent, int fedId, const Word64* trailer, Errors& errors) override;

  bool checkHeader(bool& errorsInEvent, int fedId, const Word64* header, Errors& errors) override;

  bool checkTrailer(bool& errorsInEvent, int fedId, unsigned int nWords, const Word64* trailer, Errors& errors) override;

  bool checkROC(bool& errorsInEvent, int fedId, const SiPixelFrameConverter* converter, 
		const SiPixelFedCabling* theCablingTree,
		Word32& errorWord, Errors& errors) override;



  void conversionError(int fedId, const SiPixelFrameConverter* converter, 
		       int status, Word32& errorWord, Errors& errors) override;

private:

  bool includeErrors;

  cms_uint32_t errorDetId(const SiPixelFrameConverter* converter, 
	 	          int errorType, const Word32 & word) const override;

};

#endif
