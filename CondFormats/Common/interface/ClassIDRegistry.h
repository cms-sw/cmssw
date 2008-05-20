#ifndef CondFormatCommon_ClassIDRegistry_H
#define CondFormatCommon_ClassIDRegistry_H
#include <typeinfo>
#include<vector>
#include<string>

namespace cond {
  class ClassIDRegistry {
  public:
    vector<std::string> sids;
    vector<const char*> csids;
    ClassIDRegistry(char const* pfix);
    struct Elem {
      ClassIDRegistry * registry;
      Elem() {
	// magic: works only if a file local registry exists in the file
	registry = &packageClassIDRegistry;
      }
      virtual std::string id()=0; 
      void registerMe(const std::type_info& t);
    }
      static std::string id(const std::type_info& t);

  private:
    std::string prefix;

  };

  template<typename T>
  struct ClassID : public  ClassIDRegistry::Elem {
    ClassID() { registerMe(typeid(T));}
  };



#endif
