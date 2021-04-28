#ifndef EventFilter_SiPixelRawToDigi_interface_ErrorCheckerPhase0_h
#define EventFilter_SiPixelRawToDigi_interface_ErrorCheckerPhase0_h
/** \class ErrorCheckerPhase0
 *
 *  
 */

#include "DataFormats/SiPixelDigi/interface/SiPixelDigiConstants.h"
#include "EventFilter/SiPixelRawToDigi/interface/ErrorCheckerBase.h"
#include "FWCore/Utilities/interface/typedefs.h"

class ErrorCheckerPhase0 : public ErrorCheckerBase {
public:
  typedef std::vector<SiPixelRawDataError> DetErrors;
  typedef std::map<cms_uint32_t, DetErrors> Errors;

  ErrorCheckerPhase0();

  bool checkROC(bool& errorsInEvent,
                int fedId,
                const SiPixelFrameConverter* converter,
                const SiPixelFedCabling* theCablingTree,
                Word32& errorWord,
                Errors& errors) override;

private:
  bool includeErrors_;

  cms_uint32_t errorDetId(const SiPixelFrameConverter* converter, int errorType, const Word32& word) const override;
};

#endif  // EventFilter_SiPixelRawToDigi_interface_ErrorCheckerPhase0_h
