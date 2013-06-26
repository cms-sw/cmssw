#ifndef PixelTrimOverride_h
#define PixelTrimOverride_h
//
// This class provide a base class for the
// pixel trim data for the pixel FEC configuration
// This is a pure interface (abstract class) that
// needs to have an implementation.
//
// Need to figure out what is 'VMEcommand' below! 
//
// All applications should just use this 
// interface and not care about the specific
// implementation
//
#include <string>
#include "PixelTrimBase.h"
#include "PixelMaskBase.h"

namespace pos{
  class PixelTrimOverride: public PixelTrimOverrideBase {

  public:

    PixelTrimOverride(std::string filename);

    //Build the commands needed to configure ROC
    //Need to use the mask bits also for this
    std::string getConfigCommand(PixelMaskBase& pixelMask);

  private:

    //need to store the private data here...

  };
}
#endif
