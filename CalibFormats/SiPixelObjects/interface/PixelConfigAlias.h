#ifndef PixelConfigAlias_h
#define PixelConfigAlias_h
/*! \file CalibFormats/SiPixelObjects/interface/PixelConfigAlias.h
*   \brief This class implements the steps that are used in a scan over Threshold and CalDelay
*
*   A longer explanation will be placed here later
*/

#include <string>
#include <vector>

namespace pos{
/*! \class PixelConfigAlias PixelConfigAlias.h "interface/PixelConfigAlias.h"
*
*   A longer explanation will be placed here later
*/
  class PixelConfigAlias {

  public:
    PixelConfigAlias(std::string name, unsigned int key) { 
      name_=name;
      key_=key;
    }
    void addVersionAlias(std::string path, std::string alias) {
      std::pair<std::string,std::string> apair(path,alias);
      versionAliases_.push_back(apair);
    }

    std::string name() { return name_; }
    unsigned int key() { return key_; }

    unsigned int nVersionAliases() { return versionAliases_.size(); }
    std::string versionAliasesPath(unsigned int i) { return versionAliases_[i].first; }
    std::string versionAliasesAlias(unsigned int i) { return versionAliases_[i].second; }
  
    void setKey(unsigned int key) {key_=key;}

    std::vector<std::pair<std::string,std::string> > versionAliases() {
      return versionAliases_;
    }


  private:

    std::string name_;
    unsigned int key_;
    std::vector<std::pair<std::string,std::string> > versionAliases_;

  };
}
#endif
