
/*

This file defines functions used to check if ROOT dictionaries
are missing for those types that require dictionaries.
Also there is a utility function named public_base_classes that
is used to find the base classes of classes used as elements in
container products. These base classes are needed to setup the
product lookup tables to support Views. That function also checks
for dictionaries of that contained class and its base classes as
it finds them.

As of this writing, the dictionary checking functions are used
in the following circumstances:

1. All produced products.

2. All products in the main ProductRegistry and that are present
in the input.

3. All consumed product types. Also for consumed element types (used by
View). But for consumed element types there is only a requirement that
the element type and its base classes have dictionaries (because there
is no way to know if the containing product type is transient or not).

4. Products declared as kept by an output module.

Transient classes are an exception to the above requirements. For
transients classes the only classes that are required to have dictionaries
are the top level type and its wrapped type. Also if it is a container
that can be accessed by Views, then its contained type and the base classes
of that contained type must also have dictionaries.  But only that.
Other contituents types of a transient type are not required to have
dictionaries. This special treatment of transients is genuinely needed
because there are multiple transient types in CMSSW which do not have
dictionaries for many of their constituent types.

For persistent types it checks the unwrapped type, the wrapped type, and
all the constituent types. It uses the TClass::GetMissingDictionaries
function from ROOT to check constituent types and depends on that.
(Currently, there is a JIRA ticket submitted related to bugs in that
ROOT function, JIRA-8208. We are trying to use the ROOT function for
that instead of creating our own CMS specific code that we need to
develop and maintain.). For transient types, TClass::GetMissingDictionaries
is not used because it requires too many of the constituent types
to have dictionaries.

*/

#include "FWCore/Reflection/interface/DictionaryTools.h"

#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Reflection/interface/BaseWithDict.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Reflection/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/WrappedClassName.h"

#include "TClass.h"
#include "TClassEdit.h"
#include "THashTable.h"

#include <algorithm>
#include <sstream>

namespace edm {

  bool checkDictionary(std::vector<std::string>& missingDictionaries, TypeID const& typeID) {
    TClass::GetClass(typeID.typeInfo());
    if (!hasDictionary(typeID.typeInfo())) {
      // a second attempt to load
      TypeWithDict::byName(typeID.className());
    }
    if (!hasDictionary(typeID.typeInfo())) {
      missingDictionaries.emplace_back(typeID.className());
      return false;
    }
    return true;
  }

  bool checkDictionaryOfWrappedType(std::vector<std::string>& missingDictionaries, TypeID const& unwrappedTypeID) {
    std::string wrappedName = wrappedClassName(unwrappedTypeID.className());
    TypeWithDict wrappedTypeWithDict = TypeWithDict::byName(wrappedName);
    return checkDictionary(missingDictionaries, wrappedName, wrappedTypeWithDict);
  }

  bool checkDictionaryOfWrappedType(std::vector<std::string>& missingDictionaries, std::string const& unwrappedName) {
    std::string wrappedName = wrappedClassName(unwrappedName);
    TypeWithDict wrappedTypeWithDict = TypeWithDict::byName(wrappedName);
    return checkDictionary(missingDictionaries, wrappedName, wrappedTypeWithDict);
  }

  bool checkDictionary(std::vector<std::string>& missingDictionaries,
                       std::string const& name,
                       TypeWithDict const& typeWithDict) {
    if (!bool(typeWithDict) || typeWithDict.invalidTypeInfo()) {
      missingDictionaries.emplace_back(name);
      return false;
    }
    return true;
  }

  bool checkClassDictionaries(std::vector<std::string>& missingDictionaries, TypeID const& typeID) {
    // For a class type with a dictionary the TClass* will be
    // non-null and hasDictionary will return true.
    // For a type like "int", the TClass* pointer will be a
    // nullptr and hasDictionary will return true.
    // For a class type without a dictionary it is possible for
    // TClass* to be non-null and hasDictionary to return false.

    TClass* tClass = TClass::GetClass(typeID.typeInfo());
    if (!hasDictionary(typeID.typeInfo())) {
      // a second attempt to load
      TypeWithDict::byName(typeID.className());
      tClass = TClass::GetClass(typeID.typeInfo());
    }
    if (!hasDictionary(typeID.typeInfo())) {
      missingDictionaries.emplace_back(typeID.className());
      return false;
    }

    if (tClass == nullptr) {
      return true;
    }

    bool result = true;

    THashTable hashTable;
    bool recursive = true;
    tClass->GetMissingDictionaries(hashTable, recursive);

    for (auto const& item : hashTable) {
      TClass const* cl = static_cast<TClass const*>(item);
      missingDictionaries.emplace_back(cl->GetName());
      result = false;
    }
    return result;
  }

  bool checkClassDictionaries(std::vector<std::string>& missingDictionaries,
                              std::string const& name,
                              TypeWithDict const& typeWithDict) {
    if (!bool(typeWithDict) || typeWithDict.invalidTypeInfo()) {
      missingDictionaries.emplace_back(name);
      return false;
    }

    TClass* tClass = typeWithDict.getClass();
    if (tClass == nullptr) {
      missingDictionaries.emplace_back(name);
      return false;
    }

    THashTable hashTable;
    bool recursive = true;
    tClass->GetMissingDictionaries(hashTable, recursive);

    bool result = true;

    for (auto const& item : hashTable) {
      TClass const* cl = static_cast<TClass const*>(item);
      missingDictionaries.emplace_back(cl->GetName());
      result = false;
    }
    return result;
  }

  void addToMissingDictionariesException(edm::Exception& exception,
                                         std::vector<std::string>& missingDictionaries,
                                         std::string const& context) {
    std::sort(missingDictionaries.begin(), missingDictionaries.end());
    missingDictionaries.erase(std::unique(missingDictionaries.begin(), missingDictionaries.end()),
                              missingDictionaries.end());

    std::ostringstream ostr;
    for (auto const& item : missingDictionaries) {
      ostr << "  " << item << "\n";
    }
    exception << "No data dictionary found for the following classes:\n\n"
              << ostr.str() << "\n"
              << "Most likely each dictionary was never generated, but it may\n"
              << "be that it was generated in the wrong package. Please add\n"
              << "(or move) the specification \'<class name=\"whatever\"/>\' to\n"
              << "the appropriate classes_def.xml file along with any other\n"
              << "information needed there. For example, if this class has any\n"
              << "transient members, you need to specify them in classes_def.xml.\n"
              << "Also include the class header in classes.h\n";

    if (!context.empty()) {
      exception.addContext(context);
    }
  }

  void throwMissingDictionariesException(std::vector<std::string>& missingDictionaries, std::string const& context) {
    std::vector<std::string> empty;
    throwMissingDictionariesException(missingDictionaries, context, empty);
  }

  void throwMissingDictionariesException(std::vector<std::string>& missingDictionaries,
                                         std::string const& context,
                                         std::vector<std::string>& producedTypes) {
    edm::Exception exception(errors::DictionaryNotFound);
    addToMissingDictionariesException(exception, missingDictionaries, context);

    if (!producedTypes.empty()) {
      std::sort(producedTypes.begin(), producedTypes.end());
      producedTypes.erase(std::unique(producedTypes.begin(), producedTypes.end()), producedTypes.end());

      std::ostringstream ostr;
      for (auto const& item : producedTypes) {
        ostr << "  " << item << "\n";
      }
      exception << "\nA type listed above might or might not be the same as a\n"
                << "type declared by a producer module with the function \'produces\'.\n"
                << "Instead it might be the type of a data member, base class,\n"
                << "wrapped type, or other object needed by a produced type. Below\n"
                << "is some additional information which lists the types declared\n"
                << "to be produced by a producer module that are associated with\n"
                << "the types whose dictionaries were not found:\n\n"
                << ostr.str() << "\n";
    }
    throw exception;
  }

  void throwMissingDictionariesException(std::vector<std::string>& missingDictionaries,
                                         std::string const& context,
                                         std::vector<std::string>& producedTypes,
                                         std::vector<std::string>& branchNames,
                                         bool fromStreamerSource) {
    edm::Exception exception(errors::DictionaryNotFound);
    addToMissingDictionariesException(exception, missingDictionaries, context);

    if (!producedTypes.empty()) {
      std::sort(producedTypes.begin(), producedTypes.end());
      producedTypes.erase(std::unique(producedTypes.begin(), producedTypes.end()), producedTypes.end());

      std::ostringstream ostr;
      for (auto const& item : producedTypes) {
        ostr << "  " << item << "\n";
      }
      if (fromStreamerSource) {
        exception << "\nA type listed above might or might not be the same as a\n"
                  << "type stored in the Event. Instead it might be the type of\n"
                  << "a data member, base class, wrapped type, or other object\n"
                  << "needed by a stored type. Below is some additional information\n"
                  << "which lists the stored types associated with the types whose\n"
                  << "dictionaries were not found:\n\n"
                  << ostr.str() << "\n";
      } else {
        exception << "\nA type listed above might or might not be the same as a\n"
                  << "type stored in the Event (or Lumi or Run). Instead it might\n"
                  << "be the type of a data member, base class, wrapped type, or\n"
                  << "other object needed by a stored type. Below is some additional\n"
                  << "information which lists the stored types associated with the\n"
                  << "types whose dictionaries were not found:\n\n"
                  << ostr.str() << "\n";
      }
    }

    if (!branchNames.empty()) {
      std::sort(branchNames.begin(), branchNames.end());
      branchNames.erase(std::unique(branchNames.begin(), branchNames.end()), branchNames.end());

      std::ostringstream ostr;
      for (auto const& item : branchNames) {
        ostr << "  " << item << "\n";
      }
      if (fromStreamerSource) {
        exception << "Missing dictionaries are associated with these branch names:\n\n" << ostr.str() << "\n";
      } else {
        exception << "Missing dictionaries are associated with these branch names:\n\n"
                  << ostr.str() << "\n"
                  << "If you do not need these branches and they are not produced\n"
                  << "in the current process, an alternate solution to adding\n"
                  << "dictionaries is to drop these branches on input using the\n"
                  << "inputCommands parameter of the PoolSource.";
      }
    }
    throw exception;
  }

  void throwMissingDictionariesException(std::vector<std::string>& missingDictionaries,
                                         std::string const& context,
                                         std::set<std::string>& producedTypes,
                                         bool consumedWithView) {
    edm::Exception exception(errors::DictionaryNotFound);
    addToMissingDictionariesException(exception, missingDictionaries, context);

    if (!producedTypes.empty()) {
      std::ostringstream ostr;
      for (auto const& item : producedTypes) {
        ostr << "  " << item << "\n";
      }
      if (consumedWithView) {
        exception << "\nThe list of types above was generated while checking for\n"
                  << "dictionaries related to products declared to be consumed\n"
                  << "using a View. They will be either the type or a base class\n"
                  << "of the type declared in a consumes declaration as the template\n"
                  << "parameter of a View. Below is some additional information\n"
                  << "which lists the type of the template parameter of the View.\n"
                  << "(It will be the same type unless the missing dictionary is\n"
                  << "for a base type):\n\n"
                  << ostr.str() << "\n";
      } else {
        exception << "\nThe list of types above was generated while checking for\n"
                  << "dictionaries related to products declared to be consumed.\n"
                  << "A type listed above might or might not be a type declared\n"
                  << "to be consumed. Instead it might be the type of a data member,\n"
                  << "base class, wrapped type or other object needed by a consumed\n"
                  << "type.  Below is some additional information which lists\n"
                  << "the types declared to be consumed by a module and which\n"
                  << "are associated with the types whose dictionaries were not\n"
                  << "found:\n\n"
                  << ostr.str() << "\n";
      }
    }
    throw exception;
  }

  bool public_base_classes(std::vector<std::string>& missingDictionaries,
                           TypeID const& typeID,
                           std::vector<TypeID>& baseTypes) {
    if (!checkDictionary(missingDictionaries, typeID)) {
      return false;
    }
    TypeWithDict typeWithDict(typeID.typeInfo());

    if (!typeWithDict.isClass()) {
      return true;
    }

    // No need to check into base classes of standard library
    // classes.
    if (TClassEdit::IsStdClass(typeWithDict.name().c_str())) {
      return true;
    }

    TypeBases bases(typeWithDict);
    bool returnValue = true;
    for (auto const& basex : bases) {
      BaseWithDict base(basex);
      if (!base.isPublic()) {
        continue;
      }
      TypeWithDict baseRflxType = base.typeOf();
      if (!checkDictionary(missingDictionaries, baseRflxType.name(), baseRflxType)) {
        returnValue = false;
        continue;
      }
      TypeID baseType{baseRflxType.typeInfo()};
      // Check to make sure this base appears only once in the
      // inheritance hierarchy.
      if (!search_all(baseTypes, baseType)) {
        // Save the type and recursive look for its base types
        baseTypes.push_back(baseType);
        if (!public_base_classes(missingDictionaries, baseType, baseTypes)) {
          returnValue = false;
          continue;
        }
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
    return returnValue;
  }

}  // namespace edm
