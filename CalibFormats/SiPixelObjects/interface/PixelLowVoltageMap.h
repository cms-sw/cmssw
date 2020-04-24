#ifndef PixelLowVoltageMap_h
#define PixelLowVoltageMap_h
/**
*   \file CalibFormats/SiPixelObjects/interface/PixelLowVoltageMap.h
*   \brief This class implements..
*
*   A longer explanation will be placed here later
*/

#include <vector>
#include <set>
#include <map>
#include <utility>
#include <string>
#include "CalibFormats/SiPixelObjects/interface/PixelConfigBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelModuleName.h"
#include "CalibFormats/SiPixelObjects/interface/PixelHdwAddress.h"
#include "CalibFormats/SiPixelObjects/interface/PixelNameTranslation.h"
#include "CalibFormats/SiPixelObjects/interface/PixelROCStatus.h"

namespace pos{
/*!  \ingroup ConfigurationObjects "Configuration Objects"
*    
*  @{
*
*  \class PixelLowVoltageMap PixelLowVoltageMap.h
*  \brief This is the documentation about PixelLowVoltageMap...
*
*   A longer explanation will be placed here later
*/
  class PixelLowVoltageMap: public PixelConfigBase {

  public:

    PixelLowVoltageMap(std::vector< std::vector < std::string> > &tableMat);
    PixelLowVoltageMap(std::string filename);

    void writeASCII(std::string dir="") const override;
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

    std::string dpNameIana(const PixelModuleName& module) const;
    std::string dpNameIdigi(const PixelModuleName& module) const;

    std::set <unsigned int> getFEDs(PixelNameTranslation* translation) const;
    std::map <unsigned int, std::set<unsigned int> > getFEDsAndChannels(PixelNameTranslation* translation) const;

  private:
    //ugly... FIXME
    std::map<PixelModuleName, std::pair<std::string, std::pair<std::string, std::string> > > dpNameMap_;
    //                                    base                    Iana          Idigi 
  };
}
/* @} */
#endif
