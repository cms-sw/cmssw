//
// This class reads the LTC configuration file
//
//
//
 
#include "CalibFormats/SiPixelObjects/interface/PixelLTCConfig.h"
#include <cassert>   

using namespace pos;
using namespace std;

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

    //ltcConfigPath_ = filename;
    string line;
    while (!in.eof()) {
       getline (in,line);
       ltcConfigStream_ << line << endl;
    }

} 

void PixelLTCConfig::writeASCII(std::string dir) const {

  if (dir!="") dir+="/";
  std::string filename=dir+"LTCConfiguration.txt";
  std::ofstream out(filename.c_str());

  //std::ifstream in(ltcConfigPath_.c_str());
  //assert(in.good());

  string configstr = ltcConfigStream_.str();

  out << configstr << endl;

  out.close();

}

   
 
