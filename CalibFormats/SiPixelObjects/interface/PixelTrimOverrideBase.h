#ifndef PixelTrimOverrideBase_h
#define PixelTrimOverrideBase_h
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
#include "CalibFormats/SiPixelObjects/interface/PixelMaskBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelConfigBase.h"

namespace pos {
  class PixelTrimOverrideBase : public PixelConfigBase {
  public:
    PixelTrimOverrideBase(std::string description, std::string creator, std::string date);

    ~PixelTrimOverrideBase() override;

    //Build the commands needed to configure ROC
    //Need to use the mask bits also for this
    virtual std::string getConfigCommand(PixelMaskBase& pixelMask) = 0;

  private:
  };

}  // namespace pos
#endif
