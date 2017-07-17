#ifndef FWCore_Utilities_MemberWithDict_h
#define FWCore_Utilities_MemberWithDict_h

/*----------------------------------------------------------------------

MemberWithDict:  A holder for a class member

----------------------------------------------------------------------*/


#include <string>

class TDataMember;

namespace edm {

class ObjectWithDict;
class TypeWithDict;

class MemberWithDict {
private:
  TDataMember* dataMember_;
public:
  MemberWithDict();
  explicit MemberWithDict(TDataMember*);
  explicit operator bool() const;
  std::string name() const;
  bool isArray() const;
  bool isConst() const;
  bool isPublic() const;
  bool isStatic() const;
  bool isTransient() const;
  size_t offset() const;
  TypeWithDict declaringType() const;
  TypeWithDict typeOf() const;
  ObjectWithDict get() const;
  ObjectWithDict get(ObjectWithDict const&) const;
};

} // namespace edm

#include "FWCore/Utilities/interface/ObjectWithDict.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

#endif // FWCore_Utilities_MemberWithDict_h
