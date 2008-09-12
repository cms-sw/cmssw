#ifndef PixelTrimACommon_h
#define PixelTrimACommon_h
//
// This class provide an implementation for
// pixel trim data here you use the same trim
// values for each pixel on a roc
//
// Need to figure out what is 'VMEcommand' below! 
//
// All applications should just use this 
// interface and not care about the specific
// implementation
//

#include <string>
#include <vector>
#include "CalibFormats/SiPixelObjects/interface/PixelTrimBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelMaskBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelROCTrimBits.h"
#include "CalibFormats/SiPixelObjects/interface/PixelNameTranslation.h"

namespace pos{
  class PixelTrimCommon: public PixelTrimBase {

  public:

    PixelTrimCommon(std::string filename);

    //Build the commands needed to configure ROCs
    //on control link

    void generateConfiguration(PixelFECConfigInterface* pixelFEC,
			       PixelNameTranslation* trans,
			       const PixelMaskBase& pixelMask) const;

    void writeBinary(std::string filename) const;

    void writeASCII(std::string filename) const;


  private:

    std::vector<PixelROCName> rocname_;
    std::vector<unsigned int> trimbits_;

  };
}
#endif
