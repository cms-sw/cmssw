#ifndef PixelMaskAllPixels_h
#define PixelMaskAllPixels_h
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
#include <vector>
#include <string>
#include "CalibFormats/SiPixelObjects/interface/PixelMaskBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelROCMaskBits.h"

namespace pos{
  class PixelMaskAllPixels: public PixelMaskBase {

  public:

    PixelMaskAllPixels(std::string filename);
    PixelMaskAllPixels(std::vector< std::vector<std::string> >& tableMat);
    

    void writeBinary(std::string filename) const;

    void writeASCII(std::string dir) const;

    const PixelROCMaskBits& getMaskBits(int ROCId) const;

    PixelROCMaskBits* getMaskBits(PixelROCName name);

  private:

    std::vector<PixelROCMaskBits> maskbits_;  
 
  };
}
#endif
