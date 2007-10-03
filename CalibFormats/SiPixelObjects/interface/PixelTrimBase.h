#ifndef PixelTrimBase_h
#define PixelTrimBase_h
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
#include "CalibFormats/SiPixelObjects/interface/PixelTrimOverrideBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelTrimBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelROCTrimBits.h"
#include "CalibFormats/SiPixelObjects/interface/PixelROCName.h"
#include "CalibFormats/SiPixelObjects/interface/PixelNameTranslation.h"
#include "CalibFormats/SiPixelObjects/interface/PixelFECConfigInterface.h"


namespace pos{
  class PixelTrimBase: public PixelConfigBase {

  public:

    PixelTrimBase(std::string description, 
		  std::string creator,
		  std::string date);

    virtual ~PixelTrimBase();
    
    void setOverride(PixelTrimOverrideBase* trimOverride);

    //Build the commands needed to configure ROCs
    //on control link

    virtual void generateConfiguration(PixelFECConfigInterface* pixelFEC,
				       PixelNameTranslation* trans,
				       const PixelMaskBase& pixelMask) const =0;
    virtual void writeBinary(std::string filename) const =0;

    virtual void writeASCII(std::string filename) const =0;

    virtual PixelROCTrimBits getTrimBits(int ROCId) const =0;

    virtual PixelROCTrimBits* getTrimBits(PixelROCName name)  =0;

    friend std::ostream& operator<<(std::ostream& s, const PixelTrimBase& mask);


  private:

    PixelTrimOverrideBase* trimOverride_;

  };
}
#endif
