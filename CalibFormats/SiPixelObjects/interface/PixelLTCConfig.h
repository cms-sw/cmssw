#ifndef PixelLTCConfig_h
#define PixelLTCConfig_h
/**
*   \file CalibFormats/SiPixelObjects/interface/PixelLTCConfig..h
*   \brief This class reads the LTC configuration file
*
*   A longer explanation will be placed here later
*/
 
#include <string>
#include <map>
#include <set>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include "CalibFormats/SiPixelObjects/interface/PixelConfigBase.h"


namespace pos{
/*!  \ingroup ConfigurationObjects "Configuration Objects"
*    
*  @{
*
*  \class PixelLTCConfig PixelLTCConfig.h
*  \brief This is the documentation about PixelLTCConfig...
*
*  This class reads the LTC configuration file
*/
  class PixelLTCConfig: public PixelConfigBase{
 
  public:
   
    PixelLTCConfig(std::vector< std::vector<std::string> > &tableMat) ;
    PixelLTCConfig(std::string filename);
    //std::string getLTCConfigPath() {return ltcConfigPath_;}
    std::stringstream& getLTCConfigStream() {return ltcConfigStream_;}

    void writeASCII(std::string dir) const override;
    void 	 writeXML(        pos::PixelConfigKey key, int version, std::string path) const override {;}
    void writeXMLHeader(  pos::PixelConfigKey key, 
				  int version, 
				  std::string path, 
				  std::ofstream *out,
				  std::ofstream *out1 = nullptr,
				  std::ofstream *out2 = nullptr
				  ) const override ;
    void writeXML(        std::ofstream *out,			                                    
			   	  std::ofstream *out1 = nullptr ,
			   	  std::ofstream *out2 = nullptr ) const override ;
    void writeXMLTrailer( std::ofstream *out, 
				  std::ofstream *out1 = nullptr,
				  std::ofstream *out2 = nullptr
				  ) const override ;
 
  private:
 
    //std::string ltcConfigPath_;
    std::stringstream ltcConfigStream_;

  };
}
/* @} */
#endif
