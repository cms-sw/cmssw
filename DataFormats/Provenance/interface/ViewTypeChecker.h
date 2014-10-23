#ifndef DataFormats_Provenance_ViewTypeChecker_h
#define DataFormats_Provenance_ViewTypeChecker_h

/*----------------------------------------------------------------------

Checks for "value_type" and "member_type" typedefs inside T (of Wrapper<T>).

----------------------------------------------------------------------*/

#include <typeinfo>

namespace edm {
  class ViewTypeChecker {
  public:
    ViewTypeChecker();
    virtual ~ViewTypeChecker();

    std::type_info const& valueTypeInfo() const {return valueTypeInfo_();}
    std::type_info const& memberTypeInfo() const {return memberTypeInfo_();}

   private:
    virtual std::type_info const& valueTypeInfo_() const = 0;
    virtual std::type_info const& memberTypeInfo_() const = 0;
  };
}
#endif
