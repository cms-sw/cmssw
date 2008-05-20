#ifndef CondFormatCommon_ClassIDRegistry_H
#define CondFormatCommon_ClassIDRegistry_H
#include <typeinfo>
#include<vector>
#include<string>

namespace cond {

  class ClassIDRegistry {
  public:
    std::vector<std::string> sids;
    std::vector<const char*> csids;
    ClassIDRegistry(char const* pfix);
    struct Elem {
      ClassIDRegistry * registry;
      inline Elem();
      void registerMe(const std::type_info& t);
    };
    void registerMe(const std::type_info& t);
  private:
    std::string prefix;

  };

  template<typename T>
  struct ClassID : public  ClassIDRegistry::Elem {
    ClassID() { registerMe(typeid(T));}
  };

}

/*
 * have to learn how to use macros...
// magic: works only if a file local registry exists in the file
#define ElemConstr(xx)  Elem(){registry = &xx;} 
*/

#endif
