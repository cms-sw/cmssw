#ifndef EventFilter_SiPixelRawToDigi_interface_ErrorCheckerBase_h
#define EventFilter_SiPixelRawToDigi_interface_ErrorCheckerBase_h
/** \class ErrorCheckerBase
 *
 *  
 */

#include "DataFormats/SiPixelDigi/interface/SiPixelDigiConstants.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelFormatterErrors.h"

class SiPixelFrameConverter;
class SiPixelFedCabling;

class ErrorCheckerBase {
public:
  ErrorCheckerBase();

  virtual ~ErrorCheckerBase() = default;

  void setErrorStatus(bool ErrorStatus);

  bool checkCRC(bool& errorsInEvent, int fedId, const Word64* trailer, SiPixelFormatterErrors& errors) const;

  bool checkHeader(bool& errorsInEvent, int fedId, const Word64* header, SiPixelFormatterErrors& errors) const;

  bool checkTrailer(
      bool& errorsInEvent, int fedId, unsigned int nWords, const Word64* trailer, SiPixelFormatterErrors& errors) const;

  void conversionError(int fedId,
                       const SiPixelFrameConverter* converter,
                       int status,
                       Word32& errorWord,
                       SiPixelFormatterErrors& errors) const;

  virtual bool checkROC(bool& errorsInEvent,
                        int fedId,
                        const SiPixelFrameConverter* converter,
                        const SiPixelFedCabling* theCablingTree,
                        Word32& errorWord,
                        SiPixelFormatterErrors& errors) const = 0;

protected:
  bool includeErrors_;
  int getConversionErrorTypeAndIssueLogMessage(int status, int fedId) const;
  void addErrorToCollectionDummy(int errorType, int fedId, Word64 word, SiPixelFormatterErrors& errors) const;
  virtual cms_uint32_t errorDetId(const SiPixelFrameConverter* converter, int errorType, const Word32& word) const = 0;
};

#endif  // EventFilter_SiPixelRawToDigi_interface_ErrorCheckerBase_h
