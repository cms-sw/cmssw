#ifndef PixelPortCardConfig_h
#define PixelPortCardConfig_h
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

    void         writeASCII(std::string dir="") const override;
    void 	 writeXML(        pos::PixelConfigKey key, int version, std::string path) const override {;}
    void writeXMLHeader(  pos::PixelConfigKey key, 
				  int version, 
				  std::string path, 
				  std::ofstream *out,
				  std::ofstream *out1 = nullptr,
				  std::ofstream *out2 = nullptr
				  ) const override ;
    void writeXML( 	  std::ofstream *out,			        			    
			   	  std::ofstream *out1 = nullptr ,
			   	  std::ofstream *out2 = nullptr ) const override ;
    void writeXMLTrailer( std::ofstream *out, 
				  std::ofstream *out1 = nullptr,
				  std::ofstream *out2 = nullptr
				  ) const override ;
  
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
    
    unsigned int new_PLL_CTR2_value(std::string CTR4or5, unsigned int last_CTR2) const;
    
  private:
    void fillNameToAddress();
    void fillDBToFileAddress();

    bool containsDeviceAddress(unsigned int deviceAddress) const;
    bool containsSetting(std::string settingName) const { return containsDeviceAddress(getdeviceAddressForSetting(settingName)); }

    void setAOHGain(std::string settingName, unsigned int value);
    void setDataBaseAOHGain(std::string settingName, unsigned int value);
    std::string AOHGainStringFromAOHNumber(unsigned int AOHNumber) const;

    std::string portcardname_;
 
    std::string  TKFECID_;//FEC ID string, as defined in tkfecconfig.dat
    unsigned int ringAddress_;//ring #
    unsigned int ccuAddress_;//CCU #

    unsigned int channelAddress_;//there are 8? channels on a CCU board
    std::vector < std::pair<unsigned int, unsigned int> > device_;//the address on the portcard, and the value of it
    unsigned int i2cSpeed_;//for the portcard, the slow i2c speed is 100kHz

///key used for sorting device_
    std::vector < unsigned int > key_;
    unsigned int aohcount_;
    void sortDeviceList();

    std::string type_; // fpix or bpix, used to determine setting names and addresses
  
    std::map<std::string, unsigned int> nameToAddress_; // translation from name to address, filled in by fillNameToAddress();
    std::map<std::string, std::string> nameDBtoFileConversion_; // filled by fillDBToFileAddress() ;
  };
}
/* @} */
#endif
