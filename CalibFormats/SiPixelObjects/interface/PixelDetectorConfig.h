#ifndef PixelDetectorConfig_h
#define PixelDetectorConfig_h
//
// This class specifies which detector
// components are used in the configuration
// (and eventually should specify which 
// xdaq process controlls which components).
// 
//
//
//

#include <vector>
#include <set>
#include <map>
#include <string>
#include "CalibFormats/SiPixelObjects/interface/PixelConfigBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelModuleName.h"
#include "CalibFormats/SiPixelObjects/interface/PixelHdwAddress.h"
#include "CalibFormats/SiPixelObjects/interface/PixelNameTranslation.h"
#include "CalibFormats/SiPixelObjects/interface/PixelROCStatus.h"

namespace pos{
  class PixelDetectorConfig: public PixelConfigBase {

  public:

    PixelDetectorConfig(std::vector< std::vector < std::string> > &tableMat);
    PixelDetectorConfig(std::string filename);

    unsigned int getNModules() const;

    PixelModuleName getModule(unsigned int i) const;

    const std::vector <PixelModuleName>& getModuleList() const { return modules_; }


    void writeASCII(std::string dir="") const;

    bool containsModule(const PixelModuleName& moduleToFind) const;

    std::set <unsigned int> getFEDs(PixelNameTranslation* translation) const;
    std::map <unsigned int, std::set<unsigned int> > getFEDsAndChannels(PixelNameTranslation* translation) const;

    //friend std::ostream& operator<<(std::ostream& s, const PixelDetectorconfig& config);

  private:

    std::vector<PixelModuleName> modules_;   

    std::map<PixelROCName, PixelROCStatus> rocs_;
 
  };
}
#endif
