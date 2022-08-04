#ifndef FWCore_Reflection_DictionaryTools_h
#define FWCore_Reflection_DictionaryTools_h

/*----------------------------------------------------------------------

DictionaryTools provides a small number of dictionary based tools, used in
the CMS event model.

----------------------------------------------------------------------*/

#include <set>
#include <string>
#include <vector>

class TClass;

namespace edm {

  class Exception;
  class TypeID;
  class TypeWithDict;

  bool checkDictionary(std::vector<std::string>& missingDictionaries, TypeID const& typeID);

  bool checkDictionaryOfWrappedType(std::vector<std::string>& missingDictionaries, TypeID const& unwrappedTypeID);

  bool checkDictionaryOfWrappedType(std::vector<std::string>& missingDictionaries, std::string const& unwrappedName);

  bool checkDictionary(std::vector<std::string>& missingDictionaries,
                       std::string const& name,
                       TypeWithDict const& typeWithDict);

  bool checkClassDictionaries(std::vector<std::string>& missingDictionaries, TypeID const& typeID);

  bool checkClassDictionaries(std::vector<std::string>& missingDictionaries,
                              std::string const& name,
                              TypeWithDict const& typeWithDict);

  void addToMissingDictionariesException(edm::Exception& exception,
                                         std::vector<std::string>& missingDictionaries,
                                         std::string const& context);

  void throwMissingDictionariesException(std::vector<std::string>& missingDictionaries, std::string const& context);

  void throwMissingDictionariesException(std::vector<std::string>& missingDictionaries,
                                         std::string const& context,
                                         std::vector<std::string>& producedTypes);

  void throwMissingDictionariesException(std::vector<std::string>& missingDictionaries,
                                         std::string const& context,
                                         std::vector<std::string>& producedTypes,
                                         std::vector<std::string>& branchNames,
                                         bool fromStreamerSource = false);

  void throwMissingDictionariesException(std::vector<std::string>& missingDictionaries,
                                         std::string const& context,
                                         std::set<std::string>& producedTypes,
                                         bool consumedWithView);

  bool public_base_classes(std::vector<std::string>& missingDictionaries,
                           TypeID const& typeID,
                           std::vector<TypeID>& baseTypes);
}  // namespace edm

#endif  // FWCore_Reflection_DictionaryTools_h
