#ifndef FWCore_Utilities_DictionaryTools_h
#define FWCore_Utilities_DictionaryTools_h

/*----------------------------------------------------------------------

DictionaryTools provides a small number of dictionary based tools, used in
the CMS event model.

----------------------------------------------------------------------*/

#include <string>
#include <vector>

class TClass;

namespace edm {

  class TypeID;
  class TypeWithDict;

  bool checkDictionary(std::vector<std::string>& missingDictionaries,
                       TypeID const& typeID);

  bool checkDictionaryOfWrappedType(std::vector<std::string>& missingDictionaries,
                                    TypeID const& unwrappedTypeID);

  bool checkDictionaryOfWrappedType(std::vector<std::string>& missingDictionaries,
                                    std::string const& unwrappedName);

  bool checkDictionary(std::vector<std::string>& missingDictionaries,
                       std::string const& name,
                       TypeWithDict const& typeWithDict);

  bool checkClassDictionaries(std::vector<std::string>& missingDictionaries,
                              TypeID const& typeID);

  bool checkClassDictionaries(std::vector<std::string>& missingDictionaries,
                              std::string const& name,
                              TypeWithDict const& typeWithDict);

  void throwMissingDictionariesException(std::vector<std::string>& missingDictionaries,
                                         std::string const& context);

  void throwMissingDictionariesException(std::vector<std::string>& missingDictionaries,
                                         std::string const& context,
                                         std::vector<std::string>& branchNames);

  void public_base_classes(TypeWithDict const& type,
                           std::vector<TypeWithDict>& baseTypes);
} // namespace edm

#endif // FWCore_Utilities_DictionaryTools_h
