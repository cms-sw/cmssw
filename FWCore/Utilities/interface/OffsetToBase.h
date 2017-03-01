#ifndef FWCore_Utilities_OffsetToBase_h
#define FWCore_Utilities_OffsetToBase_h
#include <typeinfo>

/*
 * For any class used in Views, RefToBase, or Ptr,
 * class template OffsetToBase must be specialized for any class
 * with multiple inheritance (i.e. with non-zero offset to
 * at least one base class).
 * A specialization would look something like this
 * (in YourClass.h"
 *
 * #include "FWCore/Utilities/interface/OffsetToBase.h"
 * namespace edm {
 *   template<>
 *   class OffsetToBase<YourClass> {
 *     public OffsetToBase() {}
 *     size_t offsetToBase(std::type_info const& baseTypeInfo) const {
 *       int const dummy = 0;
 *       YourClass const* object = reinterpret_cast<YourClass const*>(&dummy);
 *       void const* objectPtr = object;
 *       if(baseTypeInfo == typeid(BaseClass1)) {
 *          BaseClass1 const* base = object;
 *          void const* basePtr = base;
 *          return static_cast<char const*>(basePtr) - static_cast<char const*>(objectPtr);
 *       }
 *       if(baseTypeInfo == typeid(BaseClass2)) {
 *          ...
 *       }
 *       etc.
 *     }
 *   };
 * }
 *
*/


namespace edm {
  template<typename T>
  class OffsetToBase {
  public:
    OffsetToBase() {} 
    size_t offsetToBase(std::type_info const& baseTypeInfo) const {
      return 0;
    }
  };

  template<typename T>
  void const* pointerToBase(std::type_info const& baseTypeInfo, T const* address) {
    OffsetToBase<T> offsetToBase;
    int offset = offsetToBase.offsetToBase(baseTypeInfo);
    void const* ptr = address;
    return static_cast<char const*>(ptr) + offset;
  }
}
#endif
