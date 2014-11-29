#ifndef FWCore_Utilities_DictionaryTools_h
#define FWCore_Utilities_DictionaryTools_h

/*----------------------------------------------------------------------

DictionaryTools provides a small number of dictionary based tools, used in
the CMS event model.

----------------------------------------------------------------------*/

#include <set>
#include <string>
#include <vector>

namespace edm {

  class TypeWithDict;
  typedef std::set<std::string> StringSet;

  void checkDictionaries(std::string const& name, bool noComponents = false);
  void throwMissingDictionariesException();
  void loadMissingDictionaries();
  StringSet& missingTypes();
  StringSet& foundTypes();

  void public_base_classes(TypeWithDict const& type,
                           std::vector<TypeWithDict>& baseTypes);

  std::string const& dictionaryPlugInPrefix();
}

#endif
