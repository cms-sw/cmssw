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
  public:
    MemberWithDict();

    explicit MemberWithDict(TDataMember* dataMember);

    std::string name() const;

    ObjectWithDict get() const;

    ObjectWithDict get(ObjectWithDict const& obj) const;

    TypeWithDict declaringType() const;

    TypeWithDict typeOf() const;

    bool isConst() const;

    bool isPublic() const;

    bool isStatic() const;

    bool isTransient() const;

    size_t offset() const;

#ifndef __GCCXML__
    explicit operator bool() const;
#endif

  private:

    TDataMember* dataMember_;
  };

}
#endif
