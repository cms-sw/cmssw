//
// This class reads the TTC configuration file
//
//
//
 
#include "CalibFormats/SiPixelObjects/interface/PixelTTCciConfig.h"
   
PixelTTCciConfig::PixelTTCciConfig(std::string filename):
  PixelConfigBase(" "," "," "){

    std::ifstream in(filename.c_str());

    if (!in.good()){
	std::cout << "Could not open:"<<filename<<std::endl;
	assert(0);
    }
    else {
	std::cout << "Opened:"<<filename<<std::endl;
    }

    ttcConfigPath_ = filename;

} 

std::string PixelTTCciConfig::getTTCConfigPath() {
  return ttcConfigPath_;
}
   
 

