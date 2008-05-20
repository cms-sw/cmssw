#ifndef CondFormatCommon_ClassIDRegistry_H
#define CondFormatCommon_ClassIDRegistry_H
#include <typeinfo>
#include<vector>
#include<string>

namespace cond {

  namespace idCategories {
    std::string const dictIDCategory("LCGDictID");
    std::string const pythonIDCategory("CondPythonID");
  }

  class ClassIDRegistry {
  public:
    std::vector<std::string> sids;
    std::vector<const char*> csids;
    ClassIDRegistry(std::string const & pfix);
    class Elem {
    public:
      inline Elem(const std::type_info& t) : tinfo(t) {}
      inline Elem(const std::type_info& t, int);
      inline const std::type_info& type() const { return tinfo;}
    private:
      ClassIDRegistry * registry;
      const char * registerMe(const std::type_info& t);

    };
    const char * registerMe(const std::type_info& t);

  private:
    std::string prefix;
    const std::type_info& tinfo;

  };

  template<typename T>
  struct ClassID : public  ClassIDRegistry::Elem {
    ClassID() : Elem(typeid(T) {}
    ClassID(int i) : Elem(typeid(T),i) {}
  };


}

// magic: works only if a file local registry exists in the file
#define ELEM_CONSTR(xx_)  \
  cond::ClassIDRegistry::Elem::Elem(const std::type_info& t,int) : tinfo(t) {registry = &xx_;registerMe(t);} 


#include "FWCore/PluginManager/interface/PluginFactory.h"
namespace cond{
  typedef edmplugin::PluginFactory<ClassIDRegistry::Elem*() > ClassIdFactory;
  typedef ClassIDRegistry::Elem ClassInfo;

}

#endif
