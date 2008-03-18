#ifndef PixelTKFECConfig_h
#define PixelTKFECConfig_h
/**
* \file CalibFormats/SiPixelObjects/interface/PixelTKFECConfig.h
* \brief This class specifies which TKFEC boards are used and how they are addressed
*
*   A longer explanation will be placed here later
*
*/
#include <iostream>
#include <vector>
#include <string>
#include "CalibFormats/SiPixelObjects/interface/PixelConfigBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelTKFECParameters.h"

namespace pos{
/*!  \ingroup ConfigurationObjects "Configuration Objects"
*    
*  @{
*
*  \class PixelTKFECConfig PixelTKFECConfig.h
*  \brief This class specifies which TKFEC boards are used and how they are addressed
*
*   A longer explanation will be placed here later
*
*/
  class PixelTKFECConfig: public PixelConfigBase {

  public:

    PixelTKFECConfig(std::string filename);  //  <---- Modified for the conversion from parallel vectors to object that contain the configuration
   
    PixelTKFECConfig(std::vector<std::vector<std::string> >& tableMat ); 

    virtual ~PixelTKFECConfig(); 

    unsigned int getNTKFECBoards() const;

    std::string  getTKFECID(unsigned int i) const;
    unsigned int getCrate(unsigned int i) const;
    std::string  getType(unsigned int i) const;
    unsigned int getAddress(unsigned int i) const;
    unsigned int crateFromTKFECID(std::string TKFECID) const;
    std::string  typeFromTKFECID(std::string TKFECID) const;
    unsigned int addressFromTKFECID(std::string TKFECID) const;

    virtual void writeASCII(std::string dir) const;
    
  private:
    std::vector< PixelTKFECParameters > TKFECconfig_;
  };
}
/* @} */
#endif
