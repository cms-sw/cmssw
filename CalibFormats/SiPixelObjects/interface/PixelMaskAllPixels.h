#ifndef PixelMaskAllPixels_h
#define PixelMaskAllPixels_h
/**
*   \file CalibFormats/SiPixelObjects/interface/PixelMaskAllPixels..h
*   \brief This clss implements..
*
*   A longer explanation will be placed here later
*/
#include <vector>
#include <string>
#include "CalibFormats/SiPixelObjects/interface/PixelMaskBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelROCMaskBits.h"

namespace pos{
/*!  \ingroup ConfigurationObjects "Configuration Objects"
*    
*  @{
*
*  \class PixelMaskAllPixels PixelMaskAllPixels.h
*  \brief This is the documentation about PixelMaskAllPixels...
*
*/
  class PixelMaskAllPixels: public PixelMaskBase {

  public:

    PixelMaskAllPixels(std::string filename);
    PixelMaskAllPixels(std::vector< std::vector<std::string> >& tableMat);
// modified by MR on 18-04-2008 10:05:04
    PixelMaskAllPixels() ;
    void addROCMaskBits(PixelROCMaskBits);
    

    void writeBinary(std::string filename) const;

    void writeASCII(std::string dir) const;

    const PixelROCMaskBits& getMaskBits(int ROCId) const;

    PixelROCMaskBits* getMaskBits(PixelROCName name);

  private:

    std::vector<PixelROCMaskBits> maskbits_;  
 
  };
}
/* @} */
#endif
