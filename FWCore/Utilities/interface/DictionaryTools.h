#ifndef FWCore_Utilities_DictionaryTools_h
#define FWCore_Utilities_DictionaryTools_h

/*----------------------------------------------------------------------

DictionaryTools provides a small number of dictionary based tools, used in
the CMS event model.

----------------------------------------------------------------------*/

#include <set>
#include <string>
#include <vector>

#include "FWCore/Utilities/interface/TypeID.h"

namespace edm {

class TypeID;
class TypeWithDict;
using TypeSet = std::set<TypeID>;

bool checkClassDictionary(TypeID const& type);
void checkClassDictionaries(TypeID const& type, bool recursive = true);
bool checkTypeDictionary(TypeID const& type);
void checkTypeDictionaries(TypeID const& type, bool recursive = true);
void throwMissingDictionariesException();
void loadMissingDictionaries();
TypeSet& missingTypes();

void public_base_classes(TypeWithDict const& type,
                         std::vector<TypeWithDict>& baseTypes);
} // namespace edm

#endif // FWCore_Utilities_DictionaryTools_h
