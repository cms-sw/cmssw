//
// This class reads the LTC configuration file
//
//
//
 
#include "CalibFormats/SiPixelObjects/interface/PixelLTCConfig.h"
   
using namespace pos;

PixelLTCConfig::PixelLTCConfig(std::string filename):
  PixelConfigBase(" "," "," "){

    std::ifstream in(filename.c_str());

    if (!in.good()){
	std::cout << "Could not open:"<<filename<<std::endl;
	assert(0);
    }
    else {
	std::cout << "Opened:"<<filename<<std::endl;
    }

    ltcConfigPath_ = filename;

} 

std::string PixelLTCConfig::getLTCConfigPath() {
  return ltcConfigPath_;
}
   
 
