#ifndef COND_DBCommon_ClassInfoLoader_H
#define COND_DBCommon_ClassInfoLoader_H

/**
 * loads plugin corresponding to token....
 */

#include<boost/shared_ptr.hpp>
#include<string>

namespace ROOT {
  namespace Reflex {
    class Type;
  }

}

namespace cond {

  class ClassInfo;

  // decode token
  std::string classID(std::string const & token);

  // find and return
  boost::shared_ptr<ClassInfo> classInfo(std::string const & token);

  // load plugin (delete classInfo)
  bool loadClassByToken(std::string const & token);

  ROOT::Reflex::Type const & reflexTypeByToken(std::string const & token);

}


#endif // COND_DBCommon_ClassInfoLoader_H
