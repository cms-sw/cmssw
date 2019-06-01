#ifndef ErrorCheckerBase_H
#define ErrorCheckerBase_H
/** \class ErrorCheckerBase
 *
 *  
 */

#include "DataFormats/SiPixelRawData/interface/SiPixelRawDataError.h"

#include <vector>
#include <map>


class SiPixelFrameConverter;
class SiPixelFedCabling;

class ErrorCheckerBase {

public:
  typedef cms_uint32_t Word32;
  typedef cms_uint64_t Word64;

  typedef std::vector<SiPixelRawDataError> DetErrors;
  typedef std::map<cms_uint32_t, DetErrors> Errors;

  virtual ~ErrorCheckerBase() {};

  virtual void setErrorStatus(bool ErrorStatus)=0;

  virtual bool checkCRC(bool& errorsInEvent, int fedId, const Word64* trailer, Errors& errors)=0;

  virtual bool checkHeader(bool& errorsInEvent, int fedId, const Word64* header, Errors& errors)=0;

  virtual bool checkTrailer(bool& errorsInEvent, int fedId, unsigned int nWords, const Word64* trailer, Errors& errors)=0;

  virtual bool checkROC(bool& errorsInEvent, int fedId, const SiPixelFrameConverter* converter, 
		const SiPixelFedCabling* theCablingTree,
		Word32& errorWord, Errors& errors)=0;



  virtual void conversionError(int fedId, const SiPixelFrameConverter* converter, 
		       int status, Word32& errorWord, Errors& errors)=0;

private:

  virtual  cms_uint32_t errorDetId(const SiPixelFrameConverter* converter, 
	 	          int errorType, const Word32 & word) const =0;

};

#endif
