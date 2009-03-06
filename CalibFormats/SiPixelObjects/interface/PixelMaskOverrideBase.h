#ifndef PixelMaskOverrideBase_h
#define PixelMaskOverrideBase_h
//
// This class provide a base class for the
// pixel mask data for the pixel FEC configuration
// This is a pure interface (abstract class) that
// needs to have an implementation.
//
// All applications should just use this 
// interface and not care about the specific
// implementation
//
//

#include "CalibFormats/SiPixelObjects/interface/PixelROCMaskBits.h"
#include "CalibFormats/SiPixelObjects/interface/PixelConfigBase.h"

namespace pos{
  class PixelMaskOverrideBase: public PixelConfigBase {

  public:

    PixelMaskOverrideBase(std::string description, 
			  std::string creator,
			  std::string date);

    virtual ~PixelMaskOverrideBase();

    virtual PixelROCMaskBits getMaskBits(int ROCId)=0;

  private:

  };
}
#endif
