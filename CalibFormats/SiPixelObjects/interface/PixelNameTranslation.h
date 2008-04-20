#ifndef PixelNameTranslation_h
#define PixelNameTranslation_h
//
// This class provides a translation from
// the naming documents standard to specify
// the ROC to the corresponding set of
// mfec, mfecchanner, hubaddress portadd and rocid
//
//

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

  class PixelNameTranslation: public PixelConfigBase {

  public:
 
    PixelNameTranslation(std::vector< std::vector<std::string> > &tableMat);
    PixelNameTranslation(std::string filename);

    virtual ~PixelNameTranslation(){}

    // Probably these functions should never be used, and instead we should call similar functions in PixelDetectorConfig.
    std::list<const PixelROCName*> getROCs() const;
    std::list<const PixelModuleName*> getModules() const;
    std::set<PixelChannel> getChannels() const; // returns all channels
    std::set<PixelChannel> getChannels(const PixelDetectorConfig& aDetectorConfig) const; // only returns channels on modules found in the detector config

    const PixelHdwAddress* getHdwAddress(const PixelROCName& aROC) const;

    //Should really use a different type of hdw address for a channel
    const PixelHdwAddress& getHdwAddress(const PixelChannel& aChannel) const;
    const PixelHdwAddress& firstHdwAddress(const PixelModuleName& aModule) const;
    
    const PixelChannel& getChannelForROC(const PixelROCName& aROC) const;
    std::set< PixelChannel > getChannelsOnModule(const PixelModuleName& aModule) const;
    
    friend std::ostream& operator<<(std::ostream& s, const PixelNameTranslation& table);

    std::vector<PixelROCName> getROCsFromFEDChannel(unsigned int fednumber, 
						    unsigned int fedchannel) const;

    PixelROCName ROCNameFromFEDChannelROC(unsigned int fednumber, 
					  unsigned int channel,
					  unsigned int roc) const;
					  
    bool ROCNameFromFEDChannelROCExists(unsigned int fednumber, 
					unsigned int channel,
					unsigned int roc) const;
					  
    PixelChannel ChannelFromFEDChannel(unsigned int fednumber, unsigned int fedchannel) const;
					  
    std::vector<PixelROCName> getROCsFromChannel(const PixelChannel& aChannel) const;
    std::vector<PixelROCName> getROCsFromModule(const PixelModuleName& aModule) const;

    void writeASCII(std::string dir="") const;
    
    bool ROCexists(PixelROCName theROC) ; // Added by Dario
    const PixelChannel& getChannelFromHdwAddress(const PixelHdwAddress& aHdwAddress) const;

  private:
  
        
    std::map<PixelROCName,PixelHdwAddress> translationtable_;  

    // This is a bit ugly, since the PixelHdwAddress contains the ROC number, which isn't really relevant to a PixelChannel.
    std::map<PixelChannel, PixelHdwAddress > channelTranslationTable_;

  };
}
#endif
