#include "IOPool/Streamer/interface/StreamedProducts.h"
#include "IOPool/CommonService/interface/ClassFiller.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "Reflex/Type.h"
#include "PluginManager/PluginCapabilities.h"
#include "TClass.h"

#include <string>

namespace edm {

  static void loadCap(const std::string& name) {
    std::string fname("LCGReflex/");
    fname += name;
    seal::PluginCapabilities::get()->load(fname);
	
    try {
      const seal::reflex::Type cc = seal::reflex::Type::byName(name);
    } 
    catch(...) {
      std::cerr << "Error: could not find Class object for " << name << std::endl;
      return;
    }
  }

  // ---------------------

  void loadExtraClasses() {
    loadCap(std::string("edm::ProdPair"));
    loadCap(std::string("edm::SendProds"));
    loadCap(std::string("edm::SendEvent"));
    loadCap(std::string("edm::SendDescs"));
    loadCap(std::string("edm::SendJobHeader"));
    ClassFiller();
  }

  namespace {
    seal::reflex::Type const getReflectClass(std::type_info const& ti) {
      seal::reflex::Type const typ = seal::reflex::Type::byTypeInfo(ti);
      return typ;
    }

    TClass* getRootClass(std::string const& name) {
      TClass* tc = TClass::GetClass(name.c_str());    
      
      // get ROOT TClass for this product
      // CINT::Type* cint_type = CINT::Type::get(typ_ref);
      // tc_ = cint_type->rootClass();
      // TClass* tc = TClass::GetClass(typeid(se));
      // tc_ = TClass::GetClass("edm::SendEvent");
      
      if(tc == 0) {
	throw cms::Exception("Configuration","getRootClass")
	  << "could not find TClass for " << name
	  << "\n";
      }
      
      return tc;
    }
  }

  // ---------------------
  TClass* getTClass(std::type_info const& ti) {
    seal::reflex::Type const typ = getReflectClass(ti);
    return getRootClass(typ.name(seal::reflex::SCOPED));
  }
}
