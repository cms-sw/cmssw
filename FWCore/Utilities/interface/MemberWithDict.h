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

    ObjectWithDict get() const;

    ObjectWithDict get(ObjectWithDict const& obj) const;

    TypeWithDict declaringType() const;

    TypeWithDict typeOf() const;

    bool isConst() const {
      return member_.IsConst();
    }

    bool isConstructor() const {
      return member_.IsConstructor();
    }

    bool isDestructor() const {
      return member_.IsDestructor();
    }

    bool isFunctionMember() const {
      return member_.IsFunctionMember();
    }

    bool isOperator() const {
      return member_.IsOperator();
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

    size_t functionParameterSize(bool required = false) const {
      return member_.FunctionParameterSize(required);
    }

    size_t offset() const {
      return member_.Offset();
    }

    void invoke(ObjectWithDict const& obj, ObjectWithDict* ret, std::vector<void*> const& values = std::vector<void*>()) const;

    bool operator<(MemberWithDict const& other) const {
      return member_ < other.member_;
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
