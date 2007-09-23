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
#include <iostream>

#include "CalibFormats/SiPixelObjects/interface/PixelConfigBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelNameTranslation.h"
#include "CalibFormats/SiPixelObjects/interface/PixelROCName.h"
#include "CalibFormats/SiPixelObjects/interface/PixelModuleName.h"
#include "CalibFormats/SiPixelObjects/interface/PixelHdwAddress.h"

class PixelNameTranslation: public PixelConfigBase {

 public:
 
    PixelNameTranslation(std::vector< std::vector<std::string> > &tableMat);
    PixelNameTranslation(std::string filename);

    std::list<const PixelROCName*> getROCs() const;
    std::list<const PixelModuleName*> getModules() const;

    const PixelHdwAddress* getHdwAddress(const PixelROCName& aROC) const;

    //Should really use a different type of hdw address for a module
    //See comment below...
    const std::vector<PixelHdwAddress>* getHdwAddress(const PixelModuleName& aModule) const;
    
    friend std::ostream& operator<<(std::ostream& s, const PixelNameTranslation& table);

    std::vector<PixelROCName> getROCsFromFEDChannel(unsigned int fednumber, 
						    unsigned int fedchannel) const;

    PixelROCName ROCNameFromFEDChannelROC(unsigned int fednumber, 
					  unsigned int channel,
					  unsigned int roc) const;
					  
    std::vector<PixelROCName> getROCsFromModule(const PixelModuleName& aModule) const;

    void writeASCII();
    
 private:
        
    std::map<PixelROCName,PixelHdwAddress> translationtable_;  

    //FIXME This code is not really good as we should have a more
    //general solution that works for the dual TBM mode in the
    //barrel. One possible solution would be to split the PixelHdwAddress
    //to a FEC and FED piece.
    //For now I will add two PixelHdwAddresses per module name
    //to accommodate the barrel. (Will use a vector to list them..)
    std::map<PixelModuleName,std::vector<PixelHdwAddress> > moduleTranslationtable_;  
    
};

#endif
