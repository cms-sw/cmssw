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

  bool checkROC(bool& errorsInEvent,
                int fedId,
                const SiPixelFrameConverter* converter,
                const SiPixelFedCabling* theCablingTree,
                Word32& errorWord,
                Errors& errors) override;

private:
  bool includeErrors;

  cms_uint32_t errorDetId(const SiPixelFrameConverter* converter, int errorType, const Word32& word) const override;
};

#endif
