#include "IOPool/Streamer/interface/ClassFiller.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/DictionaryTools.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

#include "TClass.h"

#include <set>
#include <algorithm>
#include <iostream>

namespace edm {

  bool loadCap(std::string const& name, std::vector<std::string>& missingDictionaries) {
    FDEBUG(1) << "Loading dictionary for " << name << "\n";
    TypeWithDict typeWithDict = TypeWithDict::byName(name);
    return checkClassDictionaries(missingDictionaries, name, typeWithDict);
  }

  void doBuildRealData(std::string const& name) {
    FDEBUG(3) << "doing BuildRealData for " << name << "\n";
    TClass* ttest = TClass::GetClass(name.c_str());
    if (ttest != nullptr) {
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
      std::vector<std::string> missingDictionaries;
      loadCap(std::string("edm::StreamedProduct"), missingDictionaries);
      loadCap(std::string("std::vector<edm::StreamedProduct>"), missingDictionaries);
      loadCap(std::string("edm::SendEvent"), missingDictionaries);
      loadCap(std::string("std::vector<edm::BranchDescription>"), missingDictionaries);
      loadCap(std::string("edm::SendJobHeader"), missingDictionaries);
      if (!missingDictionaries.empty()) {
        std::string context("Calling loadExtraClasses, checking dictionaries");
        throwMissingDictionariesException(missingDictionaries, context);
      }
    }
    done=true;
  }

  namespace {
    TClass* getRootClass(std::string const& name) {
      TClass* tc = TClass::GetClass(name.c_str());    
      
      if(tc == nullptr) {
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
