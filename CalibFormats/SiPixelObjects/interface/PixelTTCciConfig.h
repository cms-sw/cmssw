#ifndef PixelTTCciConfig_h
#define PixelTTCciConfig_h
//
// This class reads the TTC configuration file
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
  class PixelTTCciConfig: public PixelConfigBase{
 
  public:
   
    PixelTTCciConfig(std::string filename);
    std::string getTTCConfigPath();

    virtual void writeASCII(std::string dir) const;
 
  private:
 
    std::string ttcConfigPath_;

  };
}
#endif
