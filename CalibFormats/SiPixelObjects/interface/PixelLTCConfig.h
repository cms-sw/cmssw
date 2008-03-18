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
   
    PixelLTCConfig(std::string filename);
    std::string getLTCConfigPath();

    virtual void writeASCII(std::string dir) const;

 
  private:
 
    std::string ltcConfigPath_;

  };
}
/* @} */
#endif
