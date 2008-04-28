#ifndef PixelTKPCIFECConfig_h
#define PixelTKPCIFECConfig_h
/**
* \file CalibFormats/SiPixelObjects/interface/PixelPortCardConfig.h
* \brief This class specifies the settings on the TKPCIFEC and the settings on the portcard
*
*   A longer explanation will be placed here later
*/
#include <vector>
#include <string>
#include <map>
#include "CalibFormats/SiPixelObjects/interface/PixelConfigBase.h"

namespace pos{

/*!  \ingroup ConfigurationObjects "Configuration Objects"
*    
*  @{
*
*  \class PixelPortCardConfig PixelPortCardConfig.h
*  \brief This is the documentation about PixelNameTranslation...
*
*   
*  This class specifies the settings on the TKPCIFEC and the settings on the portcard 
*   
*/
  class PixelPortCardConfig: public PixelConfigBase{

  public:
  
    PixelPortCardConfig(std::vector < std::vector< std::string> >  &tableMat);
    PixelPortCardConfig(std::string);

    void writeASCII(std::string dir="") const;
  
    const std::string& getPortCardName() const { return portcardname_; }
    void setPortCardName(std::string newName) { portcardname_ = newName; }
    
    unsigned int getdevicesize() const;
    std::string  getTKFECID() const;
    unsigned int getringAddress() const;
    unsigned int getccuAddress() const;
    unsigned int getchannelAddress() const;
    unsigned int geti2cSpeed() const;
    std::string  gettype() const;
    unsigned int getdeviceAddress(unsigned int i) const;
    unsigned int getdeviceValues(unsigned int i) const;
    unsigned int getdeviceAddressForSetting(std::string settingName) const;
    unsigned int getdeviceValuesForSetting(std::string settingName) const;
    unsigned int getdeviceValuesForAddress(unsigned int address) const;
    unsigned int getAOHBias(unsigned int AOHNumber) const {return getdeviceValuesForAddress(AOHBiasAddressFromAOHNumber(AOHNumber));}
    void setdeviceValues(unsigned int address, unsigned int value);
    void setdeviceValues(std::string settingName, unsigned int value);
  
    unsigned int AOHBiasAddressFromAOHNumber(unsigned int AOHNumber) const;
    unsigned int AOHGainAddressFromAOHNumber(unsigned int AOHNumber) const;
    
    void setAOHGain(unsigned int AOHNumber, unsigned int value) {setAOHGain(AOHGainStringFromAOHNumber(AOHNumber),value);}
    unsigned int getAOHGain(unsigned int AOHNumber) const;
  
  private:
    void fillNameToAddress();

    void setAOHGain(std::string settingName, unsigned int value);
    std::string AOHGainStringFromAOHNumber(unsigned int AOHNumber) const;

    std::string portcardname_;
 
    std::string  TKFECID_;//FEC ID string, as defined in tkfecconfig.dat
    unsigned int ringAddress_;//ring #
    unsigned int ccuAddress_;//CCU #

    unsigned int channelAddress_;//there are 8? channels on a CCU board
    std::vector < std::pair<unsigned int, unsigned int> > device_;//the address on the portcard, and the value of it
    unsigned int i2cSpeed_;//for the portcard, the slow i2c speed is 100kHz
  
    std::string type_; // fpix or bpix, used to determine setting names and addresses
  
    std::map<std::string, unsigned int> nameToAddress_; // translation from name to address, filled in by fillNameToAddress();
  };
}
/* @} */
#endif
