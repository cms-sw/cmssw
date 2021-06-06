#ifndef FWCore_Utilities_TypeIDBase_h
#define FWCore_Utilities_TypeIDBase_h
// -*- C++ -*-
//
// Package:     Utilities
// Class  :     TypeIDBase
//
/**\class TypeIDBase TypeIDBase.h FWCore/Utilities/interface/TypeIDBase.h

 Description: Base class for classes used to compare C++ types

 Usage:
    This class is not meant to be used polymorphically (which is why there is no virtual destructor).
 Instead it is used to hold a common methods needed by all type comparing classes.

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Nov 10 14:59:35 EST 2005
//

// system include files
#include <typeinfo>

// user include files

// forward declarations
namespace edm {
  class TypeIDBase {
  public:
    struct Def {};

    constexpr TypeIDBase() noexcept : t_(&(typeid(Def))) {}

    constexpr explicit TypeIDBase(const std::type_info& t) noexcept : t_(&t) {}

    constexpr explicit TypeIDBase(const std::type_info* t) noexcept : t_(t == nullptr ? &(typeid(Def)) : t) {}

    // ---------- const member functions ---------------------

    /** Returned C-style string owned by system; do not delete[] it.
         This is the (horrible, mangled, platform-dependent) name of
         the type. */
    const char* name() const { return t_->name(); }

    bool operator<(const TypeIDBase& b) const { return t_->before(*(b.t_)); }
    bool operator==(const TypeIDBase& b) const { return (*t_) == *(b.t_); }

  protected:
    constexpr const std::type_info& typeInfo() const { return *t_; }

  private:
    //const TypeIDBase& operator=(const TypeIDBase&); // stop default

    // ---------- member data --------------------------------
    //NOTE: since the compiler generates the type_info's and they have a lifetime
    //  good for the entire application, we do not have to delete it
    //  We also are using a pointer rather than a reference so that operator= will work
    const std::type_info* t_;
  };

  inline bool operator>(const TypeIDBase& a, const TypeIDBase& b) { return b < a; }

  inline bool operator!=(const TypeIDBase& a, const TypeIDBase& b) { return !(a == b); }

}  // namespace edm

#endif
