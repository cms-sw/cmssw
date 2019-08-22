#ifndef PixelTrimBase_h
#define PixelTrimBase_h
/**
* \file CalibFormats/SiPixelObjects/interface/PixelTrimBase.h
* \brief This class provides a base class for the pixel trim data for the pixel FEC configuration
* 
*  This is a pure interface (abstract class) that
*  needs to have an implementation.
* 
*  Need to figure out what is 'VMEcommand' below! 
* 
*  All applications should just use this 
*  interface and not care about the specific
*  implementation
*
*/

#include <string>
#include "CalibFormats/SiPixelObjects/interface/PixelTrimOverrideBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelTrimBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelROCTrimBits.h"
#include "CalibFormats/SiPixelObjects/interface/PixelROCName.h"
#include "CalibFormats/SiPixelObjects/interface/PixelNameTranslation.h"
#include "CalibFormats/SiPixelObjects/interface/PixelFECConfigInterface.h"

namespace pos {
  /*!  \defgroup TrimObjects "Trim Objects"
*    \ingroup ConfigurationObjects "Configuration Objects"
*    
*  @{
*
*  \class PixelTrimBase PixelTrimBase.h
*  \brief This class provides a base class for the pixel trim data for the pixel FEC configuration
*
*  This is a pure interface (abstract class) that
*  needs to have an implementation.
* 
*  Need to figure out what is 'VMEcommand' below! 
* 
*  All applications should just use this 
*  interface and not care about the specific
*  implementation
*
*/
  class PixelTrimBase : public PixelConfigBase {
  public:
    PixelTrimBase(std::string description, std::string creator, std::string date);

    ~PixelTrimBase() override;

    void setOverride(PixelTrimOverrideBase* trimOverride);

    //Build the commands needed to configure ROCs
    //on control link

    virtual void generateConfiguration(PixelFECConfigInterface* pixelFEC,
                                       PixelNameTranslation* trans,
                                       const PixelMaskBase& pixelMask) const = 0;
    virtual void writeBinary(std::string filename) const = 0;

    void writeASCII(std::string filename) const override = 0;
    void writeXML(pos::PixelConfigKey key, int version, std::string path) const override { ; }
    void writeXMLHeader(pos::PixelConfigKey key,
                        int version,
                        std::string path,
                        std::ofstream* out,
                        std::ofstream* out1 = nullptr,
                        std::ofstream* out2 = nullptr) const override {
      ;
    }
    void writeXML(std::ofstream* out, std::ofstream* out1 = nullptr, std::ofstream* out2 = nullptr) const override { ; }
    void writeXMLTrailer(std::ofstream* out,
                         std::ofstream* out1 = nullptr,
                         std::ofstream* out2 = nullptr) const override {
      ;
    }

    virtual PixelROCTrimBits getTrimBits(int ROCId) const = 0;

    virtual PixelROCTrimBits* getTrimBits(PixelROCName name) = 0;

    friend std::ostream& operator<<(std::ostream& s, const PixelTrimBase& mask);

  private:
    PixelTrimOverrideBase* trimOverride_;
  };
}  // namespace pos
/* @} */
#endif
