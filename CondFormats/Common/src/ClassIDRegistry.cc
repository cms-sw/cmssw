#include "CondFormats/Common/interface/ClassIDRegistry.h"
#include "StorageSvc/DbReflex.h"
#include <typeinfo>
#include "cxxabi.h"

namespace pool {  void genMD5(const std::string& s, void* code);  }

namespace cond {

  namespace {
    std::string id(const std::type_info& t) {
      const ROOT::Reflex::Type type=ROOT::Reflex::Type::ByTypeInfo(t);
      std::string s;
      if (type) {
	ROOT::Reflex::PropertyList pl = type.Properties();
	if ( pl.HasProperty("ClassID") )  {
	  s = pl.PropertyAsString("ClassID");
	}
	else  {
	  char buff[20];
	  pool::genMD5(type.Name(ROOT::Reflex::SCOPED),buff);
	  s = ((pool::Guid*)buff)->toString();
	}
      }
      else {
	// only for test
	int status=0;
	std::string name = __cxxabiv1::__cxa_demangle(t.name(), 0, 0, &status);
	char buff[20];
	pool::genMD5(name,buff);
	s = ((pool::Guid*)buff)->toString();
      }
      return s;
    }
  }

  ClassIDRegistry::ClassIDRegistry(char const* pfix) : prefix(pfix){}

  void ClassIDRegistry::registerMe(const std::type_info& t) {
    sids.push_back(prefix+id(t)+'/0');
    csids.push_back(sids.data());
  }
  
  void ClassIDRegistry::Elem::registerMe(const std::type_info& t) {
    registry->registerMe(t);
  }

  

}
