#include "CondCore/DBCommon/interface/ClassInfoLoader.h"
#include "CondFormats/Common/interface/ClassIDRegistry.h"

#include "FWCore/PluginManager/interface/PluginManager.h"


namespace cond {

  // decode token
  std::string classID(std::string const & token) {
    static std::string const clid("CLID=");
    std::string::size_type s = token.find(clid) + clid.size();
    std::string::size_type e = token.find(']',s);
    return token.substr(s,e-s);

  }

  // find and return
  boost::shared_ptr<ClassInfo> classInfo(std::string const & token) {
    static std::string const prefix = cond::idCategories::dictIDCategory + "/";
    std::string pluginName = prefix + classID(token);
    return boost::shared_ptr<ClassInfo>(ClassInfoFactory::get()->create(pluginName));
  }
  
  // load plugin (delete classInfo)
  bool loadClassByToken(std::string const & token) {
     boost::shared_ptr<ClassInfo> ci = classInfo(token);
     return ci.get()!=0;
  }

}
