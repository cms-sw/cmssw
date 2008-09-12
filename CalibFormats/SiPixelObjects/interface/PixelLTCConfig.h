#ifndef PixelLTCConfig_h
#define PixelLTCConfig_h
//
// This class reads the LTC configuration file
//
//
//
 
#include <string>
#include <map>
#include <set>
#include <fstream>
#include <iostream>
#include "CalibFormats/SiPixelObjects/interface/PixelConfigBase.h"


namespace pos{
  class PixelLTCConfig: public PixelConfigBase{
 
  public:
   
    PixelLTCConfig(std::string filename);
    std::string getLTCConfigPath();

    virtual void writeASCII(std::string dir) const;

 
  private:
 
    std::string ltcConfigPath_;

  };
}
#endif
