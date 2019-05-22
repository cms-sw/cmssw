#ifndef PixelDACSettings_h
#define PixelDACSettings_h
/**
* \file CalibFormats/SiPixelObjects/interface/PixelDACSettings.h
*   \brief This class provide a base class for the pixel ROC dac data for the pixel FEC configuration
*
*   This is a pure interface (abstract class) that needs to have an implementation.
*/

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
#include "CalibFormats/SiPixelObjects/interface/PixelDetectorConfig.h"

namespace pos {
  /*!  \defgroup ConfigurationObjects "Configuration Objects"
*   \brief This is the base class of all configuration objects
*    
*   A longer explanation of what a 'configuration object' actually is will be 
*   posted here once we find the time to write it....
*
*  @{
*
*  \class PixelDACSettings PixelDACSettings.h
*  \brief This class is responsible for manipulating the DACsettings of a ROC.
*
*  This is a placeholder for a lengthy description of the class, it's methods
*  behavior and additional stuff like images. This description can be arbitrary
*  long and complex, see for eg. \ref page3Sect3. <P>
*  Ut perspiciatis, unde omnis iste natus error sit voluptatem 
*  accusantium doloremque laudantium, totam rem aperiam eaque ipsa, quae ab 
*  illo inventore veritatis et quasi architecto beatae vitae dicta sunt, explicabo. 
*
*  \image html temp.png
*
*  Nemo enim ipsam voluptatem, quia voluptas sit, aspernatur aut odit aut fugit, 
*  sed quia consequuntur magni dolores eos, qui ratione voluptatem sequi nesciunt, 
*  neque porro quisquam est, qui dolorem ipsum, quia dolor sit, amet, consectetur, 
*  adipisci velit, sed quia non numquam eius modi tempora incidunt, ut labore et 
*  dolore magnam aliquam quaerat voluptatem. Ut enim ad minima veniam, quis nostrum 
*  exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi 
*  consequatur? Quis autem vel eum iure reprehenderit, qui in ea voluptate velit esse, 
*  quam nihil molestiae consequatur, vel illum, qui dolorem eum fugiat, quo 
*  voluptas nulla pariatur? 
*/

  class PixelDACSettings : public PixelConfigBase {
  public:
    PixelDACSettings(std::string filename);
    //Added by Umesh
    PixelDACSettings(std::vector<std::vector<std::string> >& tableMat);
    // modified by MR on 10-01-2008 14:47:47
    PixelDACSettings(PixelROCDACSettings& rocname);
    // modified by MR on 24-01-2008 14:28:14
    void addROC(PixelROCDACSettings& rocname);

    PixelROCDACSettings getDACSettings(int ROCId) const;
    PixelROCDACSettings* getDACSettings(PixelROCName);

    unsigned int numROCs() { return dacsettings_.size(); }

    //Generate the DAC settings
    void generateConfiguration(PixelFECConfigInterface* pixelFEC,
                               PixelNameTranslation* trans,
                               PixelDetectorConfig* detconfig,
                               bool HVon = true) const;
    void setVcthrDisable(PixelFECConfigInterface* pixelFEC, PixelNameTranslation* trans) const;
    void setVcthrEnable(PixelFECConfigInterface* pixelFEC,
                        PixelNameTranslation* trans,
                        PixelDetectorConfig* detconfig) const;

    void writeBinary(std::string filename) const;

    void writeASCII(std::string dir) const override;
    void writeXML(pos::PixelConfigKey key, int version, std::string path) const override { ; }
    void writeXMLHeader(pos::PixelConfigKey key,
                        int version,
                        std::string path,
                        std::ofstream* out,
                        std::ofstream* out1 = nullptr,
                        std::ofstream* out2 = nullptr) const override;
    void writeXML(std::ofstream* out, std::ofstream* out1 = nullptr, std::ofstream* out2 = nullptr) const override;
    void writeXMLTrailer(std::ofstream* out,
                         std::ofstream* out1 = nullptr,
                         std::ofstream* out2 = nullptr) const override;

    friend std::ostream& operator<<(std::ostream& s, const PixelDACSettings& mask);

  private:
    std::vector<PixelROCDACSettings> dacsettings_;

    bool rocIsDisabled(const PixelDetectorConfig* detconfig, const PixelROCName rocname) const;
  };
}  // namespace pos

/* @} */
#endif
