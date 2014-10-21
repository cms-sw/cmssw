#include "FWCore/Utilities/interface/DictionaryTools.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/BaseWithDict.h"
#include "FWCore/Utilities/interface/MemberWithDict.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

#include "Api.h" // for G__ClassInfo

#include "TROOT.h"
#include "TInterpreter.h"
#include "TVirtualMutex.h"

#include "boost/algorithm/string.hpp"
#include "boost/thread/tss.hpp"

#include <algorithm>
#include <sstream>

namespace edm {

  std::string const& dictionaryPlugInPrefix() {
    static std::string const prefix("LCGReflex/");
    return prefix;
  }

  static StringSet foundTypes_;
  static StringSet missingTypes_;

  bool
  find_nested_type_named(std::string const& nested_type,
                         TypeWithDict const& typeToSearch,
                         TypeWithDict& found_type) {
    // Look for a sub-type named 'nested_type'
    TypeWithDict foundType = typeToSearch.nestedType(nested_type);
    if(bool(foundType)) {
      found_type = foundType;
      return true;
    }
    return false;
  }

  bool
  is_RefVector(TypeWithDict const& possibleRefVector,
               TypeWithDict& value_type) {

    static std::string const template_name("edm::RefVector");
    static std::string const member_type("member_type");
    if(template_name == possibleRefVector.templateName()) {
      return find_nested_type_named(member_type, possibleRefVector, value_type);
    }
    return false;
  }

  bool
  is_PtrVector(TypeWithDict const& possibleRefVector,
               TypeWithDict& value_type) {

    static std::string const template_name("edm::PtrVector");
    static std::string const member_type("member_type");
    static std::string const val_type("value_type");
    if(template_name == possibleRefVector.templateName()) {
      TypeWithDict ptrType;
      if(find_nested_type_named(val_type, possibleRefVector, ptrType)) {
        return find_nested_type_named(val_type, ptrType, value_type);
      }
    }
    return false;
  }

  bool
  is_RefToBaseVector(TypeWithDict const& possibleRefVector,
                     TypeWithDict& value_type) {

    static std::string const template_name("edm::RefToBaseVector");
    static std::string const member_type("member_type");
    if(template_name == possibleRefVector.templateName()) {
      return find_nested_type_named(member_type, possibleRefVector, value_type);
    }
    return false;
  }

  namespace {

    int const oneParamArraySize = 6;
    std::string const oneParam[oneParamArraySize] = {
      "vector",
      "basic_string",
      "set",
      "list",
      "deque",
      "multiset"
    };
    int const twoParamArraySize = 3;
    std::string const twoParam[twoParamArraySize] = {
      "map",
      "pair",
      "multimap"
    };


    bool
    hasCintDictionary(std::string const& name) {
      std::auto_ptr<G__ClassInfo> ci(new G__ClassInfo(name.c_str()));
        return(ci.get() && ci->IsLoaded());
    }

    // Checks if there is a dictionary for the Type t.
    // If noComponents is false, checks members and base classes recursively.
    // If noComponents is true, checks Type t only.
    void
    checkType(TypeWithDict t, bool noComponents = false) {

      // ToType strips const, volatile, array, pointer, reference, etc.,
      // and also translates typedefs.
      // To be safe, we do this recursively until we either get a void type or the same type.
      for(TypeWithDict x(t.toType()); x != t && x.typeInfo() != typeid(void); t = x, x = t.toType()) {}

      std::string name(t.name());
      boost::trim(name);

      if(foundTypes().end() != foundTypes().find(name) || missingTypes().end() != missingTypes().find(name)) {
        // Already been processed. Prevents infinite loop.
        return;
      }

      if(name.empty() || t.isFundamental() || t.isEnum() || t.typeInfo() == typeid(void)) {
        foundTypes().insert(name);
        return;
      }

      if(!bool(t)) {
        if(hasCintDictionary(name)) {
          foundTypes().insert(name);
        } else {
          missingTypes().insert(name);
        }
        return;
      }

      foundTypes().insert(name);
      if(noComponents) return;

      if(name.find("std::") == 0) {
        if(t.isTemplateInstance()) {
          std::string::size_type n = name.find('<');
          int cnt = 0;
          if(std::find(oneParam, oneParam + oneParamArraySize, name.substr(5, n - 5)) != oneParam + oneParamArraySize) {
            cnt = 1;
          } else if(std::find(twoParam, twoParam + twoParamArraySize, name.substr(5, n - 5)) != twoParam + twoParamArraySize) {
            cnt = 2;
          }
          for(int i = 0; i < cnt; ++i) {
            checkType(t.templateArgumentAt(i));
          }
        }
      } else {
        TypeDataMembers members(t);
        for(auto const& member : members) {
          MemberWithDict m(member);
          if(!m.isTransient() && !m.isStatic()) {
            checkType(m.typeOf());
          }
        }
        {
          R__LOCKGUARD(gCINTMutex);
          TypeBases bases(t);
          for(auto const& base : bases) {
            BaseWithDict b(base);
            checkType(b.typeOf());
          }
        }
      }
    }
  } // end unnamed namespace

  StringSet& missingTypes() {
    return missingTypes_;
  }

  StringSet& foundTypes() {
    // The only purpose of this cache is to stop infinite recursion.
    // ROOT maintains its own internal cache.
    return foundTypes_;
  }

  void checkDictionaries(std::string const& name, bool noComponents) {
    TypeWithDict null;
    TypeWithDict t(TypeWithDict::byName(name));
    if(t == null) {
      if(name == std::string("void")) {
        foundTypes().insert(name);
      } else {
        missingTypes().insert(name);
      }
      return;
    }
    checkType(t, noComponents);
  }

  void throwMissingDictionariesException() {
    if(!missingTypes().empty()) {
      std::ostringstream ostr;
      for (StringSet::const_iterator it = missingTypes().begin(), itEnd = missingTypes().end();
           it != itEnd; ++it) {
        ostr << *it << "\n\n";
      }
      throw Exception(errors::DictionaryNotFound)
        << "No REFLEX data dictionary found for the following classes:\n\n"
        << ostr.str()
        << "Most likely each dictionary was never generated,\n"
        << "but it may be that it was generated in the wrong package.\n"
        << "Please add (or move) the specification\n"
        << "<class name=\"whatever\"/>\n"
        << "to the appropriate classes_def.xml file.\n"
        << "If the class is a template instance, you may need\n"
        << "to define a dummy variable of this type in classes.h.\n"
        << "Also, if this class has any transient members,\n"
        << "you need to specify them in classes_def.xml.";
    }
  }

  void loadMissingDictionaries() {
    while (!missingTypes().empty()) {
      StringSet missing(missingTypes());
      for (StringSet::const_iterator it = missing.begin(), itEnd = missing.end();
         it != itEnd; ++it) {
        try {
          gROOT->GetClass(it->c_str(), kTRUE);
        }
        // We don't want to fail if we can't load a plug-in.
        catch(...) {}
      }
      missingTypes().clear();
      for (StringSet::const_iterator it = missing.begin(), itEnd = missing.end();
         it != itEnd; ++it) {
        checkDictionaries(*it);
      }
      if (missingTypes() == missing) {
        break;
      }
    }
    if (missingTypes().empty()) {
      return;
    }
    throwMissingDictionariesException();
  }

  void public_base_classes(TypeWithDict const& typeID,
                           std::vector<TypeWithDict>& baseTypes) {

    TypeWithDict type(typeID.typeInfo());
    if(type.isClass()) {
      R__LOCKGUARD(gCINTMutex);
      TypeBases bases(type);
      for(auto const& basex : bases) {
        BaseWithDict base(basex);
        if(base.isPublic()) {

          TypeWithDict baseRflxType = base.typeOf();
          if(bool(baseRflxType)) {
            TypeWithDict baseType(baseRflxType.typeInfo()); 

            // Check to make sure this base appears only once in the
            // inheritance heirarchy.
            if(!search_all(baseTypes, baseType)) {
              // Save the type and recursive look for its base types
              baseTypes.push_back(baseType);
              public_base_classes(baseType, baseTypes);
            }
            // For now just ignore it if the class appears twice,
            // After some more testing we may decide to uncomment the following
            // exception.
            /*
            else {
              throw Exception(errors::UnimplementedFeature)
                << "DataFormats/Common/src/DictionaryTools.cc in function public_base_classes.\n"
                << "Encountered class that has a public base class that appears\n"
                << "multiple times in its inheritance heirarchy.\n"
                << "Please contact the EDM Framework group with details about\n"
                << "this exception. It was our hope that this complicated situation\n"
                << "would not occur. There are three possible solutions. 1. Change\n"
                << "the class design so the public base class does not appear multiple\n"
                << "times in the inheritance heirarchy. In many cases, this is a\n"
                << "sign of bad design. 2. Modify the code that supports Views to\n"
                << "ignore these base classes, but not supply support for creating a\n"
                << "View of this base class. 3. Improve the View infrastructure to\n"
                << "deal with this case. Class name of base class: " << baseType.Name() << "\n\n";
            }
            */
          }
        }
      }
    }
  }
}
