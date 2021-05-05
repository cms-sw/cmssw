#ifndef ErrorCheckerBase_H
#define ErrorCheckerBase_H
/** \class ErrorCheckerBase
 *
 *  
 */

#include "DataFormats/SiPixelDigi/interface/SiPixelDigiConstants.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelRawDataError.h"

#include <vector>
#include <map>

class SiPixelFrameConverter;
class SiPixelFedCabling;

class ErrorCheckerBase {
public:
  typedef std::vector<SiPixelRawDataError> DetErrors;
  typedef std::map<cms_uint32_t, DetErrors> Errors;
  ErrorCheckerBase();

  virtual ~ErrorCheckerBase() = default;

  virtual void setErrorStatus(bool ErrorStatus);

  virtual bool checkCRC(bool& errorsInEvent, int fedId, const Word64* trailer, Errors& errors);

  virtual bool checkHeader(bool& errorsInEvent, int fedId, const Word64* header, Errors& errors);

  virtual bool checkTrailer(bool& errorsInEvent, int fedId, unsigned int nWords, const Word64* trailer, Errors& errors);

  virtual bool checkROC(bool& errorsInEvent,
                        int fedId,
                        const SiPixelFrameConverter* converter,
                        const SiPixelFedCabling* theCablingTree,
                        Word32& errorWord,
                        Errors& errors) = 0;

  virtual void conversionError(
      int fedId, const SiPixelFrameConverter* converter, int status, Word32& errorWord, Errors& errors);

private:
  bool includeErrors;
  virtual cms_uint32_t errorDetId(const SiPixelFrameConverter* converter, int errorType, const Word32& word) const = 0;
};

#endif
