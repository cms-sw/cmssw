#ifndef CondCoreDBCommon_ClassID_H
#define CondCoreDBCommon_ClassID_H
#include <typeinfo>
#include<vector>
#include<string>

namespace cond {

  namespace idCategories {
    std::string const pythonIDCategory("CondPythonID");
  }



  char const * className(const std::type_info& t);


  class ClassInfo {
  public:
    virtual ~ClassInfo(){}
    inline ClassInfo(const std::type_info& t) : tinfo(t) {}
    //inline ClassInfo(const std::type_info& t, int);
    inline const std::type_info& type() const { return tinfo;}
    std::string pluginName(std::string const & prefix) const;
    virtual std::string resource() const=0;
  private:
    //ClassIDRegistry * registry;
    //const char * registerMe(const std::type_info& t);
    const std::type_info& tinfo;
  };
  
  
  /**class ClassIDRegistry {
  public:
    typedef ClassInfo Elem;
    std::vector<std::string> sids;
    std::vector<const char*> csids;
    ClassIDRegistry(std::string const & pfix);
    
    const char * registerMe(const std::type_info& t);
    
  private:
    std::string prefix;
    
  };
  **/

  template<typename T>
  struct ClassID : public  ClassInfo {
    ClassID() : ClassInfo(typeid(T)) {}
    ClassID(char const * res ) : ClassInfo(typeid(T)), m_res(res) {}
//    ClassID(int i) : ClassInfo(typeid(T),i) {}
    virtual std::string resource() const { return m_res;}
  private:
    std::string m_res;
  };
  
    
}

/**
// magic: works only if a file local registry exists in the file
#define ELEM_CONSTR(xx_)						\
  cond::ClassInfo::ClassInfo(const std::type_info& t,int) : tinfo(t) {registry = &xx_;registerMe(t);} 
**/

#include "FWCore/PluginManager/interface/PluginFactory.h"
namespace cond{
  typedef edmplugin::PluginFactory<ClassInfo*() > ClassInfoFactory;
}

#define PYTHON_ID(type_, plugName_, uname_)					\
  namespace pythonID { struct EDM_PLUGIN_SYM(plugName_, uname_)  : public cond::ClassID<type_> { EDM_PLUGIN_SYM(plugName_ , uname_) () : cond::ClassID<type_>(::plugName_){}};} \
  DEFINE_EDM_PLUGIN(cond::ClassInfoFactory, EDM_PLUGIN_SYM(pythonID::plugName_, uname_) , EDM_PLUGIN_SYM(pythonID::plugName_, uname_)().pluginName(cond::idCategories::pythonIDCategory).c_str() )


#endif //  CondCoreDBCommon_ClassID_H
