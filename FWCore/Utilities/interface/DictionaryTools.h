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

bool checkClassDictionary(TypeID const& type, TypeSet& missingTypes);
void checkClassDictionaries(TypeID const& type, TypeSet& missingTypes, bool recursive = true);
bool checkTypeDictionary(TypeID const& type, TypeSet& missingTypes);
void checkTypeDictionaries(TypeID const& type, TypeSet& missingTypes, bool recursive = true);
void throwMissingDictionariesException(TypeSet const&);
void loadMissingDictionaries(TypeSet missingTypes);

void public_base_classes(TypeWithDict const& type,
                         std::vector<TypeWithDict>& baseTypes);
} // namespace edm

#endif // FWCore_Utilities_DictionaryTools_h
