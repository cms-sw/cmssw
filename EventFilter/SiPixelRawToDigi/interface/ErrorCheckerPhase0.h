#ifndef EventFilter_SiPixelRawToDigi_interface_ErrorCheckerPhase0_h
#define EventFilter_SiPixelRawToDigi_interface_ErrorCheckerPhase0_h
/** \class ErrorCheckerPhase0
 *
 *  
 */

#include "DataFormats/SiPixelDigi/interface/SiPixelDigiConstants.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelFormatterErrors.h"
#include "EventFilter/SiPixelRawToDigi/interface/ErrorCheckerBase.h"

class ErrorCheckerPhase0 : public ErrorCheckerBase {
public:
  ErrorCheckerPhase0();

  bool checkROC(bool& errorsInEvent,
                int fedId,
                const SiPixelFrameConverter* converter,
                const SiPixelFedCabling* theCablingTree,
                Word32& errorWord,
                SiPixelFormatterErrors& errors) const override;

protected:
  cms_uint32_t errorDetId(const SiPixelFrameConverter* converter, int errorType, const Word32& word) const override;
};

#endif  // EventFilter_SiPixelRawToDigi_interface_ErrorCheckerPhase0_h
