#ifndef PixelMaskOverride_h
#define PixelMaskOverride_h
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

#include <string>
#include "PixelROCMaskBits.h"

namespace pos{
  class PixelMaskOverride: public PixelMaskOverrideBase {

  public:

    PixelMaskOverride(std::string filename);

    PixelROCMaskBits getMaskBits(int ROCId );

  private:

    //need to store the input here....

  };
}
#endif
