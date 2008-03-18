#ifndef PixelPortcardMap_h
#define PixelPortcardMap_h
/**
* \file CalibFormats/SiPixelObjects/interface/PixelPortcardMap.h
* \brief This class provides the mapping between portcards and the modules controlled by the card
*
*   A longer explanation will be placed here later
*
*/
 
#include <string>
#include <vector>
#include <map>
#include <set>
#include "CalibFormats/SiPixelObjects/interface/PixelConfigBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelModuleName.h"
#include "CalibFormats/SiPixelObjects/interface/PixelTBMChannel.h"
#include "CalibFormats/SiPixelObjects/interface/PixelChannel.h"

namespace pos{
/*!  \ingroup ConfigurationObjects "Configuration Objects"
*    
*  @{
*
*  \class PixelPortCardConfig PixelPortCardConfig.h
*  \brief This is the documentation about PixelNameTranslation...
*
*  This class provides the mapping between portcards and the modules controlled by the card
*   
*/
  class PixelPortcardMap: public PixelConfigBase
  {
  public:

    PixelPortcardMap(std::string filename);

    PixelPortcardMap(std::vector< std::vector < std::string> > &tableMat);

    virtual ~PixelPortcardMap();

    // Get the port card and AOH associated with this module.  If the module has one(two) channels, this vector contains one(two) element(s).
    const std::set< std::pair< std::string, int > > PortCardAndAOHs(const PixelModuleName& aModule) const;
    //                            portcardname, aoh #

    const std::set< std::string > portcards(const PixelModuleName& aModule) const;

    int numChannels(const PixelModuleName& aModule) {return PortCardAndAOHs(aModule).size();}

    const std::pair< std::string, int > PortCardAndAOH(const PixelModuleName& aModule, const std::string& TBMChannel) const;
    const std::pair< std::string, int > PortCardAndAOH(const PixelModuleName& aModule, const PixelTBMChannel& TBMChannel) const;
    const std::pair< std::string, int > PortCardAndAOH(const PixelChannel& aChannel) const;

    // set of all modules attached to a port card
    std::set< PixelModuleName > modules(std::string portCardName) const;

    // all port cards in the map
    std::set< std::string > portcards();

    virtual void writeASCII(std::string dir) const;

  private:
    //                               portcardname, AOH #
    std::map< PixelChannel, std::pair<std::string, int> > map_;
  };
}
/* @} */
#endif
