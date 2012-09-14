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

    Reflex::Member member_;
  };

}
#endif
