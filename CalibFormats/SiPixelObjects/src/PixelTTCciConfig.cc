//
// This class reads the TTC configuration file
//
//
//
 
#include "CalibFormats/SiPixelObjects/interface/PixelTTCciConfig.h"
#include <cassert>
  
using namespace pos;
using namespace std;
 
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

void PixelTTCciConfig::writeASCII(std::string dir) const {

  
  if (dir!="") dir+="/";
  std::string filename=dir+"TTCciConfiguration.txt";
  std::ofstream out(filename.c_str());

  std::ifstream in(ttcConfigPath_.c_str());
  assert(in.good());

  string line;
  while (!in.eof()) {
    getline (in,line);
    out << line << endl;
  }

  out.close();



}


std::string PixelTTCciConfig::getTTCConfigPath() {
  return ttcConfigPath_;
}
   
 

