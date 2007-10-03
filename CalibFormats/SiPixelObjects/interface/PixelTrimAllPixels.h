#ifndef PixelTrimAllPixels_h
#define PixelTrimAllPixels_h
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
#include <vector>
#include "CalibFormats/SiPixelObjects/interface/PixelTrimBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelMaskBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelROCTrimBits.h"
#include "CalibFormats/SiPixelObjects/interface/PixelROCName.h"
#include "CalibFormats/SiPixelObjects/interface/PixelNameTranslation.h"

namespace pos{
  class PixelTrimAllPixels: public PixelTrimBase {

  public:

    PixelTrimAllPixels(std::string filename);
    PixelTrimAllPixels(std::vector<std::vector<std::string> > &tableMat);  

    //Build the commands needed to configure ROCs
    //on control link

    void generateConfiguration(PixelFECConfigInterface* pixelFEC,
			       PixelNameTranslation* trans,
			       const PixelMaskBase& pixelMask) const;

    void writeBinary(std::string filename) const;

    void writeASCII(std::string filename) const;

    PixelROCTrimBits getTrimBits(int ROCId) const;

    PixelROCTrimBits* getTrimBits(PixelROCName name);


  private:

    std::vector<std::string> rocname_;
    std::vector<PixelROCTrimBits> trimbits_;

  };
}
#endif
