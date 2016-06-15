
/*

Mostly this contains functions used to check if dictionaries
are missing for those classes that require dictionaries.
Also there is a utility function that used to find the
base classes of types used as elements in container products
that are needed to setup the product lookup tables.

As of this writing, the dictionary checking functions are used
in the following circumstances:

1. All produced products.

2. All products in the main ProductRegistry and that are present
in the input.

3. All consumed products and also for types used as element types
in a consumes request for a View.

4. Products declared as kept by an output module.

In most cases it checks the unwrapped type, the wrapped type, and
all the constituent types. It uses the TClass::GetMissingDictionaries
function from ROOT to check constituent types and depends on that.
(Currently, there is a JIRA ticket submitted related to bugs in that
ROOT function, JIRA-8208. We are trying to use the ROOT function for
that instead of creating our own CMS specific code that we need to
develop and maintain.).  The two exceptions are:

  For kept produced types, it only checks the unwrapped top level
type. These are checked later as they are also present and in
the ProductRegistry.

  For consumed element types, only the element type and its
constituents are checked because the actual product type
containing the element is not known at that point.

*/

#include "FWCore/Utilities/interface/DictionaryTools.h"

#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/BaseWithDict.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/WrappedClassName.h"

#include "TClass.h"
#include "THashTable.h"

#include <algorithm>
#include <sstream>

namespace edm {

  bool
  checkDictionary(std::vector<std::string>& missingDictionaries,
                  TypeID const& typeID) {

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

  bool checkDictionaryOfWrappedType(std::vector<std::string>& missingDictionaries,
                                    TypeID const& unwrappedTypeID) {
    std::string wrappedName = wrappedClassName(unwrappedTypeID.className());
    TypeWithDict wrappedTypeWithDict = TypeWithDict::byName(wrappedName);
    return checkDictionary(missingDictionaries, wrappedName, wrappedTypeWithDict);
  }

  bool checkDictionaryOfWrappedType(std::vector<std::string>& missingDictionaries,
                                    std::string const& unwrappedName) {
    std::string wrappedName = wrappedClassName(unwrappedName);
    TypeWithDict wrappedTypeWithDict = TypeWithDict::byName(wrappedName);
    return checkDictionary(missingDictionaries, wrappedName, wrappedTypeWithDict);
  }

  bool
  checkDictionary(std::vector<std::string>& missingDictionaries,
                  std::string const& name,
                  TypeWithDict const& typeWithDict) {
    if (!bool(typeWithDict) || typeWithDict.invalidTypeInfo()) {
      missingDictionaries.emplace_back(name);
      return false;
    }
    return true;
  }

  bool
  checkClassDictionaries(std::vector<std::string>& missingDictionaries,
                         TypeID const& typeID) {

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

    for(auto const& item : hashTable) {
      TClass const* cl = static_cast<TClass const*>(item);
      missingDictionaries.emplace_back(cl->GetName());
      result = false;
    }
    return result;
  }

  bool
  checkClassDictionaries(std::vector<std::string>& missingDictionaries,
                         std::string const& name,
                         TypeWithDict const& typeWithDict) {
    if (!bool(typeWithDict) || typeWithDict.invalidTypeInfo()) {
      missingDictionaries.emplace_back(name);
      return false;
    }

    TClass *tClass = typeWithDict.getClass();
    if (tClass == nullptr) {
      missingDictionaries.emplace_back(name);
      return false;
    }

    THashTable hashTable;
    bool recursive = true;
    tClass->GetMissingDictionaries(hashTable, recursive);

    bool result = true;

    for(auto const& item : hashTable) {
      TClass const* cl = static_cast<TClass const*>(item);
      missingDictionaries.emplace_back(cl->GetName());
      result = false;
    }
    return result;
  }

  void throwMissingDictionariesException(std::vector<std::string>& missingDictionaries,
                                         std::string const& context) {
    std::vector<std::string> branchNames;
    throwMissingDictionariesException(missingDictionaries, context, branchNames);
  }

  void throwMissingDictionariesException(std::vector<std::string>& missingDictionaries,
                                         std::string const& context,
                                         std::vector<std::string>& branchNames) {

    std::sort(missingDictionaries.begin(), missingDictionaries.end());
    missingDictionaries.erase(std::unique(missingDictionaries.begin(), missingDictionaries.end()), missingDictionaries.end());

    edm::Exception ex(errors::DictionaryNotFound);
    if (missingDictionaries.empty()) {
      return;
    } else {
      std::ostringstream ostr;
      for(auto const& item : missingDictionaries) {
        ostr << "  " << item << "\n";
      }
      ex  << "No data dictionary found for the following classes:\n\n"
          << ostr.str() << "\n"
          << "Most likely each dictionary was never generated,\n"
          << "but it may be that it was generated in the wrong package.\n"
          << "Please add (or move) the specification\n"
          << "<class name=\"whatever\"/>\n"
          << "to the appropriate classes_def.xml file.\n"
          << "Also include the class header in classes.h\n"
          << "If the class is a template instance, you may need\n"
          << "to define a dummy variable of this type in classes.h.\n"
          << "Also, if this class has any transient members,\n"
          << "you need to specify them in classes_def.xml.";
    }
    if (!branchNames.empty()) {

      std::sort(branchNames.begin(), branchNames.end());
      branchNames.erase(std::unique(branchNames.begin(), branchNames.end()), branchNames.end());

      std::ostringstream ostr;
      for(auto const& item : branchNames) {
        ostr << "  " << item << "\n";
      }
      ex  << "\n\nMissing dictionaries are associated with the following branch names:\n\n"
          << ostr.str() << "\n"
          << "If you do not need these branches, an alternate solution to\n"
          << "adding dictionaries is to drop these branches on input\n"
          << "using the inputCommands parameter of the PoolSource.";
    }
    if (!context.empty()) {
      ex.addContext(context);
    }
    throw ex;
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
