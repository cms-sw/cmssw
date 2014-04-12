#ifndef PixelTTCciConfig_h
#define PixelTTCciConfig_h
//
// This class reads the TTC configuration file
//
//
//
 
#include <string>
#include <vector>
#include <map>
#include <set>
#include <fstream>
#include <iostream>
#include <sstream>
#include "CalibFormats/SiPixelObjects/interface/PixelConfigBase.h"

namespace pos{
  class PixelTTCciConfig: public PixelConfigBase{
 
  public:
   
    PixelTTCciConfig(std::string filename);
    PixelTTCciConfig(std::vector<std::vector<std::string> > &) ;
    //std::string getTTCConfigPath() {return ttcConfigPath_;}
    std::stringstream& getTTCConfigStream() {return ttcConfigStream_;}

    virtual void writeASCII(std::string dir) const;
    virtual void writeXML(        pos::PixelConfigKey key, int version, std::string path) const {;}
    virtual void writeXMLHeader(  pos::PixelConfigKey key, 
				  int version, 
				  std::string path, 
				  std::ofstream *out,
				  std::ofstream *out1 = NULL,
				  std::ofstream *out2 = NULL
				  ) const ;
    virtual void writeXML( 	  std::ofstream *out,			        			    
			   	  std::ofstream *out1 = NULL ,
			   	  std::ofstream *out2 = NULL ) const ;
    virtual void writeXMLTrailer( std::ofstream *out, 
				  std::ofstream *out1 = NULL,
				  std::ofstream *out2 = NULL
				  ) const ;
 
  private:
 
    //std::string ttcConfigPath_;
    std::stringstream ttcConfigStream_;

  };
}
#endif
