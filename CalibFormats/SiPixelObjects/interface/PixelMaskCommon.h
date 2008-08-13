#ifndef PixelMaskCommon_h
#define PixelMaskCommon_h
//
// This class provide an implementation for
// pixel mask data where each pixel have the 
// same mask.
//
//
//

#include "CalibFormats/SiPixelObjects/interface/PixelMaskBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelROCName.h"

namespace pos{
  class PixelMaskAllPixels: public PixelMaskBase {

  public:

    PixelMaskCommon(std::string filename);

    void writeBinary(std::string filename) const;

    void writeASCII(std::string filename) const;

  private:

    std::vector<PixeROCName> rocname_;  
    std::vector<bool> maskbits_;  
 
  };
}
#endif
