#ifndef CondFormatCommon_ClassIDRegistry_H
#define CondFormatCommon_ClassIDRegistry_H
#include <typeinfo>
#include<vector>
#include<string>

namespace cond {

  namespace idCategories {
    std::string const dictIDCategory("LCGClassID");
    std::string const pythonIDCategory("CondPythonID");
  }

  class ClassIDRegistry {
  public:
    std::vector<std::string> sids;
    std::vector<const char*> csids;
    ClassIDRegistry(std::string const & pfix);
    struct Elem {
      ClassIDRegistry * registry;
      inline Elem(){}
      inline Elem(const std::type_info& t);
      const char * registerMe(const std::type_info& t);
    };
    const char * registerMe(const std::type_info& t);
  private:
    std::string prefix;

  };

  template<typename T>
  struct ClassID : public  ClassIDRegistry::Elem {
    ClassID() {}
    ClassID(int) : Elem(typeid(T)) {}
  };

}

// magic: works only if a file local registry exists in the file
#define ELEM_CONSTR(xx_)  \
cond::ClassIDRegistry::Elem::Elem(const std::type_info& t){registry = &xx_;registerMe(t);} 


#include "FWCore/PluginManager/interface/PluginFactory.h"
namespace cond{
  typedef edmplugin::PluginFactory<ClassIDRegistry::Elem*() > ClassIdFactory;
}

#endif
