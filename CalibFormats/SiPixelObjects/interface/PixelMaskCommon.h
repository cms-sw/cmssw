#ifndef PixelMaskCommon_h
#define PixelMaskCommon_h
/**
* \file CalibFormats/SiPixelObjects/interface/PixelMaskCommon.h
* \brief  This class provide an implementation for
*         pixel mask data where each pixel have the 
*         same mask.
*
*   A longer explanation will be placed here later
*
*/

#include "CalibFormats/SiPixelObjects/interface/PixelMaskBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelROCName.h"

namespace pos{
/*! \class PixelMaskAllPixels PixelMaskAllPixels.h "interface/PixelMaskAllPixels.h"
*   \brief This class implements..
*
*   A longer explanation will be placed here later
*/
  class PixelMaskAllPixels: public PixelMaskBase {

  public:

    PixelMaskCommon(std::string filename);

    void writeBinary(std::string filename) const;

    void writeASCII(std::string filename) const;
    void 	 writeXML(      pos::PixelConfigKey key, int version, std::string path)                     const {;}
    virtual void writeXMLHeader(pos::PixelConfigKey key, int version, std::string path, std::ofstream *out) const {;}
    virtual void writeXML(                                                              std::ofstream *out) const {;}
    virtual void writeXMLTrailer(                                                       std::ofstream *out) const {;}

  private:

    std::vector<PixeROCName> rocname_;  
    std::vector<bool> maskbits_;  
 
  };
}
#endif
