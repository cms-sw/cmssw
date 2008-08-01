#ifndef PixelDACSettings_h
#define PixelDACSettings_h
//
// This class provide a base class for the
// pixel ROC dac data for the pixel FEC configuration
// This is a pure interface (abstract class) that
// needs to have an implementation.
//
//
//

#include <vector>
#include <string>
#include "CalibFormats/SiPixelObjects/interface/PixelConfigBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelROCDACSettings.h"
#include "CalibFormats/SiPixelObjects/interface/PixelFECConfigInterface.h"
#include "CalibFormats/SiPixelObjects/interface/PixelROCName.h"
#include "CalibFormats/SiPixelObjects/interface/PixelNameTranslation.h"

namespace pos{
  class PixelDACSettings: public PixelConfigBase {

  public:

    PixelDACSettings(std::string filename);
    //Added by Umesh
    PixelDACSettings(std::vector<std::vector<std::string> >& tableMat);   
    // modified by MR on 10-01-2008 14:47:47
    PixelDACSettings(PixelROCDACSettings &rocname);
    // modified by MR on 24-01-2008 14:28:14
    void addROC(PixelROCDACSettings &rocname);
    
    PixelROCDACSettings getDACSettings(int ROCId) const;
    PixelROCDACSettings* getDACSettings(PixelROCName);

    unsigned int numROCs() {return dacsettings_.size();}

    //Generate the DAC settings
    void generateConfiguration(PixelFECConfigInterface* pixelFEC,
	                       PixelNameTranslation* trans) const;

    void writeBinary(std::string filename) const;

    void writeASCII(std::string dir) const;

    friend std::ostream& operator<<(std::ostream& s, const PixelDACSettings& mask);

  private:

    std::vector<PixelROCDACSettings> dacsettings_;

  };
}
#endif
