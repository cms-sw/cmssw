#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/DictionaryTools.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

#include "TClass.h"

#include <string>
#include <set>
#include <algorithm>
#include <iostream>

namespace edm {
  void loadType(TypeID const& type) {
    checkClassDictionaries(type, true);
    if (!missingTypes().empty()) {
      TypeSet missing = missingTypes();
      missingTypes().clear();
      for_all(missing, loadType);
    }
  }

  void loadCap(std::string const& name) {
    FDEBUG(1) << "Loading dictionary for " << name << "\n";
    TypeWithDict typedict = TypeWithDict::byName(name);
    if (!typedict) {
      throw cms::Exception("DictionaryMissingClass") << "The dictionary of class '" << name << "' is missing!";
    }
    TClass* cl = TClass::GetClass(name.c_str());
    loadType(TypeID(*cl->GetTypeInfo()));
  }

  void doBuildRealData(std::string const& name) {
    FDEBUG(3) << "doing BuildRealData for " << name << "\n";
    TClass* ttest = TClass::GetClass(name.c_str());
    if (ttest != 0) {
      ttest->BuildRealData();
    } else {
      throw edm::Exception(errors::Configuration)
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
    done=true;
  }

  namespace {
    TClass* getRootClass(std::string const& name) {
      TClass* tc = TClass::GetClass(name.c_str());    
      
      // get ROOT TClass for this product
      // CINT::Type* cint_type = CINT::Type::get(typ_ref);
      // tc_ = cint_type->rootClass();
      // TClass* tc = TClass::GetClass(typeid(se));
      // tc_ = TClass::GetClass("edm::SendEvent");
      
      if(tc == 0) {
	throw edm::Exception(errors::Configuration,"getRootClass")
	  << "could not find TClass for " << name
	  << "\n";
      }
      
      return tc;
    }
  }

  // ---------------------
  TClass* getTClass(std::type_info const& ti) {
    TypeID const type(ti);
    return getRootClass(type.className());
  }
}
