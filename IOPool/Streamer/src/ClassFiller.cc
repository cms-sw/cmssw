#include "DataFormats/Streamer/interface/StreamedProducts.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/ReflexTools.h"
#include "Cintex/Cintex.h"
#include "FWCore/PluginManager/interface/PluginCapabilities.h"


#include "TClass.h"
#include "G__ci.h"

#include <string>
#include <set>
#include <algorithm>

namespace edm {
  std::string getName(ROOT::Reflex::Type& cc) {
    return cc.Name(ROOT::Reflex::SCOPED);
  }

  void loadCap(std::string const& name) {
    static std::string const fname("LCGReflex/");
    FDEBUG(1) << "Loading dictionary for " << name << "\n";
    edmplugin::PluginCapabilities::get()->load(fname + name);
    checkDictionaries(name);
    if (!missingTypes().empty()) {
      StringSet missing = missingTypes();
      missingTypes().clear();
      for_all(missing, loadCap);
    }
  }

  void doBuildRealData(std::string const& name) {
    FDEBUG(3) << "doing BuildRealData for " << name << "\n";
    ROOT::Reflex::Type cc = ROOT::Reflex::Type::ByName(name);
    TClass* ttest = TClass::GetClass(getName(cc).c_str());
    if (ttest != 0) {
      ttest->BuildRealData();
    } else {
      throw cms::Exception("Configuration")
			<< "Could not find TClass for " << name << "\n";
    }
  }
  // ---------------------

  void loadExtraClasses() {
    static bool done = false;
    if (done == false) {
	loadCap(std::string("edm::StreamedProduct"));
	loadCap(std::string("std::vector<edm::StreamedProduct>"));
	loadCap(std::string("edm::SendEvent"));
	loadCap(std::string("std::vector<edm::BranchDescription>"));
	loadCap(std::string("edm::SendJobHeader"));
    }
    G__SetCatchException(0);
    ROOT::Cintex::Cintex::Enable();
    done=true;
  }

  namespace {
    ROOT::Reflex::Type const getReflectClass(std::type_info const& ti) {
      ROOT::Reflex::Type const typ = ROOT::Reflex::Type::ByTypeInfo(ti);
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
    ROOT::Reflex::Type const typ = getReflectClass(ti);
    return getRootClass(typ.Name(ROOT::Reflex::SCOPED));
  }
}
