#ifndef PixelNameTranslation_h
#define PixelNameTranslation_h
/**
* \file CalibFormats/SiPixelObjects/interface/PixelNameTranslation.h
* \brief This class provides a translation from the naming documents standard to specify
*        the ROC to the corresponding set of
*        mfec, mfecchanner, hubaddress portadd and rocid
*
*   A longer explanation will be placed here later
*/

#include <map>
#include <string>
#include <vector>
#include <list>
#include <set>
#include <iostream>

#include "CalibFormats/SiPixelObjects/interface/PixelConfigBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelNameTranslation.h"
#include "CalibFormats/SiPixelObjects/interface/PixelROCName.h"
#include "CalibFormats/SiPixelObjects/interface/PixelModuleName.h"
#include "CalibFormats/SiPixelObjects/interface/PixelChannel.h"
#include "CalibFormats/SiPixelObjects/interface/PixelHdwAddress.h"

namespace pos{

  class PixelDetectorConfig;

/*!  \ingroup ConfigurationObjects "Configuration Objects"
*    
*  @{
*
*  \class PixelNameTranslation PixelNameTranslation.h
*  \brief This is the documentation about PixelNameTranslation...
*
*   This class provides a translation from the naming documents standard to specify
*   the ROC to the corresponding set of
*   mfec, mfecchanner, hubaddress portadd and rocid
*/
  class PixelNameTranslation: public PixelConfigBase {

  public:
 
    PixelNameTranslation(std::vector< std::vector<std::string> > &tableMat);
    PixelNameTranslation(std::string filename);

    ~PixelNameTranslation() override{}

    // Probably these functions should never be used, and instead we should call similar functions in PixelDetectorConfig.
    std::list<const PixelROCName*> getROCs() const;
    std::list<const PixelModuleName*> getModules() const;
    std::set<PixelChannel> getChannels() const; // returns all channels
    std::set<PixelChannel> getChannels(const PixelDetectorConfig& aDetectorConfig) const; // only returns channels on modules found in the detector config

    const PixelHdwAddress* getHdwAddress(const PixelROCName& aROC) const;

    //Should really use a different type of hdw address for a channel
    const PixelHdwAddress& getHdwAddress(const PixelChannel& aChannel) const;
    const PixelHdwAddress& firstHdwAddress(const PixelModuleName& aModule) const;
    
    const bool checkFor(const PixelROCName& aROC) const ; 

    // Added for Debbie (used there only) to allow integrity checks (Dario)
    bool checkROCExistence(const PixelROCName& aROC) const ;
 
    const PixelChannel& getChannelForROC(const PixelROCName& aROC) const;
    std::set< PixelChannel > getChannelsOnModule(const PixelModuleName& aModule) const;
    
    friend std::ostream& operator<<(std::ostream& s, const PixelNameTranslation& table);

    const std::vector<PixelROCName>& getROCsFromFEDChannel(unsigned int fednumber, 
						     unsigned int fedchannel) const;
    
    PixelROCName ROCNameFromFEDChannelROC(unsigned int fednumber, 
					  unsigned int channel,
					  unsigned int roc) const;
					  
    bool ROCNameFromFEDChannelROCExists(unsigned int fednumber, 
					unsigned int channel,
					unsigned int roc) const;
					  
    PixelChannel ChannelFromFEDChannel(unsigned int fednumber, unsigned int fedchannel) const;

    bool FEDChannelExist(unsigned int fednumber, unsigned int fedchannel) const;
					  
    const std::vector<PixelROCName>& getROCsFromChannel(const PixelChannel& aChannel) const;
    std::vector<PixelROCName> getROCsFromModule(const PixelModuleName& aModule) const;

    void writeASCII(std::string dir="") const override;
    void 	 writeXML(        pos::PixelConfigKey key, int version, std::string path)       	    const override   ;
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
    
    bool ROCexists(PixelROCName theROC) ; // Added by Dario
    const PixelChannel& getChannelFromHdwAddress(const PixelHdwAddress& aHdwAddress) const;


    std::map <unsigned int, std::set<unsigned int> > getFEDsAndChannels() const;

  private:
  
        
    std::map<PixelROCName,PixelHdwAddress> translationtable_;  

    std::map<PixelHdwAddress, PixelROCName, PixelHdwAddress> fedlookup_;  

    // This is a bit ugly, since the PixelHdwAddress contains the ROC number, which isn't really relevant to a PixelChannel.
    std::map<PixelChannel, PixelHdwAddress > channelTranslationTable_;
    std::map<PixelHdwAddress, PixelChannel > hdwTranslationTable_;

    std::map<unsigned int, std::map<unsigned int, std::vector<PixelROCName> > > rocsFromFEDidAndChannel_;
    //       FED id                  FED channel

    std::vector<PixelROCName> buildROCsFromFEDChannel(unsigned int fednumber, 
						      unsigned int fedchannel) const;


  };
}
/* @} */
#endif
