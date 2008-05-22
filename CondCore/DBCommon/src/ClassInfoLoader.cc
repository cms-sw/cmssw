#include "CondCore/DBCommon/interface/ClassInfoLoader.h"
#include "CondFormats/Common/interface/ClassIDRegistry.h"
#include "CondFormats/Common/interface/Exception.h"

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
     // will never return false as pluginMgr throw!
     return ci.get()!=0;
  }


  namespace {
    std::string const errmsg("Unable to load class for token ");
    std::string const orimsg(". Original error msg was ");

  }

  ROOT::Reflex::Type const & reflexTypeByToken(std::string const & token) {
    const pool::Guid guid(cond::classID(token));
    {
      // look if already loaded
      ROOT::Reflex::Type const & type = pool::DbReflex::forGuid(guid);
      if (type) return type;
    }
    try {
      // plugin mgr will throw fist: still
      if (!cond::loadClassByToken(token)) 
	throw cond::Exception(errmsg+token);
    }
    catch (cms::Exception const & e) {
      throw cond::Exception(errmsg+token+orimsg+e.what(););
    }
    ROOT::Reflex::Type const & type = pool::DbReflex::forGuid(guid);
    if (!type) throw cond::Exception(errmsg+token +". Problem with DbReflex");
    return type;
  }

}
