#ifndef PixelTrimACommon_h
#define PixelTrimACommon_h
/**
* \file CalibFormats/SiPixelObjects/interface/PixelTrimCommon.h
* \brief This class implements..
*
*   A longer explanation will be placed here later
*
*/

#include <string>
#include <vector>
#include "CalibFormats/SiPixelObjects/interface/PixelTrimBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelMaskBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelROCTrimBits.h"
#include "CalibFormats/SiPixelObjects/interface/PixelNameTranslation.h"

namespace pos{
/*!  \ingroup TrimObjects "Trim Objects"
*    \ingroup ConfigurationObjects "Configuration Objects"
*    
*  @{
*
*  \class PixelTrimCommon PixelTrimCommon.h
*  \brief This class implements..
*
*   A longer explanation will be placed here later
*
*/
  class PixelTrimCommon: public PixelTrimBase {

  public:

    PixelTrimCommon(std::string filename);

    //Build the commands needed to configure ROCs
    //on control link

    void generateConfiguration(PixelFECConfigInterface* pixelFEC,
			       PixelNameTranslation* trans,
			       const PixelMaskBase& pixelMask) const;

    void writeBinary(std::string filename) const;

    void         writeASCII(std::string filename)                                                           const  ;
    void         writeXML(      pos::PixelConfigKey key, int version, std::string path)                     const {;}
    virtual void writeXMLHeader(pos::PixelConfigKey key, int version, std::string path, std::ofstream *out) const {;}
    virtual void writeXML(                                                              std::ofstream *out) const {;}
    virtual void writeXMLTrailer(                                                       std::ofstream *out) const {;}


  private:

    std::vector<PixelROCName> rocname_;
    std::vector<unsigned int> trimbits_;

  };
}
#endif
