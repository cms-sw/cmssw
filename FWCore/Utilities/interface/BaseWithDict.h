#ifndef FWCore_Utilities_BaseWithDict_h
#define FWCore_Utilities_BaseWithDict_h

/*----------------------------------------------------------------------

BaseWithDict:  A holder for a base class

----------------------------------------------------------------------*/


#include <string>

class TBaseClass;

namespace edm {

class TypeWithDict;

class BaseWithDict {
private:
  TBaseClass* baseClass_;
public:
  BaseWithDict();
  explicit BaseWithDict(TBaseClass*);
  bool isPublic() const;
  std::string name() const;
  TypeWithDict typeOf() const;
  size_t offset() const;
};

} // namespace edm

#include "FWCore/Utilities/interface/TypeWithDict.h"

#endif // FWCore_Utilities_BaseWithDict_h
