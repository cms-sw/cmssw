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

    std::string typeName() const;

    ObjectWithDict get(ObjectWithDict const& obj) const;

    TypeWithDict returnType() const;

    bool isTransient() const {
      return member_.IsTransient();
    }

    void invoke(ObjectWithDict const& obj, ObjectWithDict* ret, std::vector<void*> const& values = std::vector<void*>()) const;

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
