#ifndef FWCore_Utilities_propagate_const_h
#define FWCore_Utilities_propagate_const_h
// -*- C++ -*-
//
// Package:     FWCore/Utilities
// Class  :     propagate_const
//
/**\class propagate_const propagate_const.h "FWCore/Utilities/interface/propagate_const.h"

 Description: propagate const to pointer like objects. Based on C++ experimental std::propagate_const.

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri, 18 Dec 2015 14:56:12 GMT
//

// system include files
#include <type_traits>
#include <utility>

// user include files

// forward declarations

namespace edm {

  template <typename T>
  class propagate_const;

  template <typename T>
  T& get_underlying(propagate_const<T>&);
  template <typename T>
  T const& get_underlying(propagate_const<T> const&);

  template <typename T>
  class propagate_const {
  public:
    friend T& get_underlying<T>(propagate_const<T>&);
    friend T const& get_underlying<T>(propagate_const<T> const&);

    using element_type = typename std::remove_reference<decltype(*std::declval<T&>())>::type;

    propagate_const() = default;

    propagate_const(propagate_const<T>&&) = default;
    propagate_const(propagate_const<T> const&) = delete;
    template <typename U>
    propagate_const(U&& iValue) : m_value(std::forward<U>(iValue)) {}

    propagate_const<T>& operator=(propagate_const&&) = default;
    propagate_const<T>& operator=(propagate_const<T> const&) = delete;

    template <typename U>
    propagate_const& operator=(U&& iValue) {
      m_value = std::forward<U>(iValue);
      return *this;
    }

    // ---------- const member functions ---------------------
    element_type const* get() const { return to_raw_pointer(m_value); }
    element_type const* operator->() const { return this->get(); }
    element_type const& operator*() const { return *m_value; }

    operator element_type const*() const { return this->get(); }

    // ---------- member functions ---------------------------
    element_type* get() { return to_raw_pointer(m_value); }
    element_type* operator->() { return this->get(); }
    element_type& operator*() { return *m_value; }

    operator element_type*() { return this->get(); }

  private:
    template <typename Up>
    static constexpr element_type* to_raw_pointer(Up* u) {
      return u;
    }

    template <typename Up>
    static constexpr element_type* to_raw_pointer(Up& u) {
      return u.get();
    }

    template <typename Up>
    static constexpr const element_type* to_raw_pointer(const Up* u) {
      return u;
    }

    template <typename Up>
    static constexpr const element_type* to_raw_pointer(const Up& u) {
      return u.get();
    }

    // ---------- member data --------------------------------
    T m_value;
  };

  template <typename T>
  T& get_underlying(propagate_const<T>& iP) {
    return iP.m_value;
  }
  template <typename T>
  T const& get_underlying(propagate_const<T> const& iP) {
    return iP.m_value;
  }

}  // namespace edm

#endif
