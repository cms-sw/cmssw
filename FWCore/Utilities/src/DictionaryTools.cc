#include "FWCore/Utilities/interface/DictionaryTools.h"

#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/BaseWithDict.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/MemberWithDict.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

#include "TClass.h"
#include "TInterpreter.h"
#include "THashTable.h"

#include "boost/algorithm/string.hpp"
#include "boost/thread/tss.hpp"

#include <algorithm>
#include <sstream>
#include <string>

namespace edm {

  static TypeSet missingTypes_;

  TypeSet&
  missingTypes() {
    return missingTypes_;
  }

  bool
  checkTypeDictionary(TypeID const& type) {
    TClass *cl = TClass::GetClass(type.typeInfo(), true);
    if(cl == nullptr) {
      // Assume not a class
      return true;
    }
    if(!cl->HasDictionary()) {
      missingTypes().insert(type);
      return false;
    }
    return true;
  }

  void
  checkTypeDictionaries(TypeID const& type, bool recursive) {
    TClass *cl = TClass::GetClass(type.typeInfo(), true);
    if(cl == nullptr) {
      // Assume not a class
      return;
    }
    THashTable result;
    cl->GetMissingDictionaries(result, recursive); 
    for(auto const& item : result) {
      TClass const* cl = static_cast<TClass const*>(item);
      missingTypes().insert(TypeID(cl->GetTypeInfo()));
    }
  }

  bool
  checkClassDictionary(TypeID const& type) {
    TClass *cl = TClass::GetClass(type.typeInfo(), true);
    if(cl == nullptr) {
      throw Exception(errors::DictionaryNotFound)
          << "No TClass for class: '" << type.className() << "'" << std::endl;
    }
    if(!cl->HasDictionary()) {
      missingTypes().insert(type);
      return false;
    }
    return true;
  }

  void
  checkClassDictionaries(TypeID const& type, bool recursive) {
    TClass *cl = TClass::GetClass(type.typeInfo(), true);
    if(cl == nullptr) {
      throw Exception(errors::DictionaryNotFound)
          << "No TClass for class: '" << type.className() << "'" << std::endl;
    }
    THashTable result;
    cl->GetMissingDictionaries(result, recursive); 
    for(auto const& item : result) {
      TClass const* cl = static_cast<TClass const*>(item);
      missingTypes().insert(TypeID(cl->GetTypeInfo()));
    }
  }

  void
  throwMissingDictionariesException() {
    if (!missingTypes().empty()) {
      std::ostringstream ostr;
      for(auto const& item : missingTypes()) {
        ostr << item << "\n\n";
      }
      throw Exception(errors::DictionaryNotFound)
          << "No data dictionary found for the following classes:\n\n"
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

  void
  loadMissingDictionaries() {
    while (!missingTypes().empty()) {
      TypeSet missing(missingTypes());
      for(auto const& item : missing) {
        try {
          TClass::GetClass(item.typeInfo(), kTRUE);
        }
        // We don't want to fail if we can't load a plug-in.
        catch (...) {}
      }
      missingTypes().clear();
      for(auto const& item : missing) {
        checkTypeDictionaries(item, true);
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

  void
  public_base_classes(TypeWithDict const& typeID,
                      std::vector<TypeWithDict>& baseTypes) {
    if (!typeID.isClass()) {
      return;
    }
    TypeWithDict type(typeID.typeInfo());
    TypeBases bases(type);
    for (auto const& basex : bases) {
      BaseWithDict base(basex);
      if (!base.isPublic()) {
        continue;
      }
      TypeWithDict baseRflxType = base.typeOf();
      if (!bool(baseRflxType)) {
        continue;
      }
      TypeWithDict baseType(baseRflxType.typeInfo());
      // Check to make sure this base appears only once in the
      // inheritance hierarchy.
      if (!search_all(baseTypes, baseType)) {
        // Save the type and recursive look for its base types
        baseTypes.push_back(baseType);
        public_base_classes(baseType, baseTypes);
      }
      // For now just ignore it if the class appears twice,
      // After some more testing we may decide to uncomment the following
      // exception.
      //
      //else {
      //  throw Exception(errors::UnimplementedFeature)
      //    << "DataFormats/Common/src/DictionaryTools.cc in function public_base_classes.\n"
      //    << "Encountered class that has a public base class that appears\n"
      //    << "multiple times in its inheritance heirarchy.\n"
      //    << "Please contact the EDM Framework group with details about\n"
      //    << "this exception. It was our hope that this complicated situation\n"
      //    << "would not occur. There are three possible solutions. 1. Change\n"
      //    << "the class design so the public base class does not appear multiple\n"
      //    << "times in the inheritance heirarchy. In many cases, this is a\n"
      //    << "sign of bad design. 2. Modify the code that supports Views to\n"
      //    << "ignore these base classes, but not supply support for creating a\n"
      //    << "View of this base class. 3. Improve the View infrastructure to\n"
      //    << "deal with this case. Class name of base class: " << baseType.Name() << "\n\n";
      //}
    }
  }

} // namespace edm
