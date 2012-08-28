#ifndef FWCore_Utilities_MemberWithDict_h
#define FWCore_Utilities_MemberWithDict_h

/*----------------------------------------------------------------------
  
MemberWithDict:  A holder for a class member

----------------------------------------------------------------------*/

#include <string>

#include "Reflex/Member.h"

namespace edm {

  class ObjectWithDict;
  class TypeWithDict;

  class MemberWithDict {
  public:
    MemberWithDict() : member_() {}

    explicit MemberWithDict(Reflex::Member const& member) : member_(member) {}

    std::string name() const {return member_.Name();}

    ObjectWithDict get() const;

    ObjectWithDict get(ObjectWithDict const& obj) const;

    TypeWithDict declaringType() const;

    TypeWithDict typeOf() const;

    bool isConst() const {
      return member_.IsConst();
    }

    bool isPublic() const {
      return member_.IsPublic();
    }

    bool isStatic() const {
      return member_.IsStatic();
    }

    bool isTransient() const {
      return member_.IsTransient();
    }

    size_t offset() const {
      return member_.Offset();
    }

#ifndef __GCCXML__
    explicit operator bool() const {
      return bool(member_);
    }
#endif

  private:

    Reflex::Member member_;
  };

}
#endif
