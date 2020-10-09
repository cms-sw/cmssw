#ifndef FWCore_Utilities_EDGetToken_h
#define FWCore_Utilities_EDGetToken_h
// -*- C++ -*-
//
// Package:     FWCore/Utilities
// Class  :     EDGetToken
//
/**\class EDGetToken EDGetToken.h "FWCore/Utilities/interface/EDGetToken.h"

 Description: A Token used to get data from the EDM

 Usage:
    A EDGetToken is created by calls to 'consumes' or 'mayConsume' from an EDM module.
 The EDGetToken can then be used to quickly retrieve data from the edm::Event, edm::LuminosityBlock or edm::Run.
 
The templated form, EDGetTokenT<T>, is the same as EDGetToken except when used to get data the framework
 will skip checking that the type being requested matches the type specified during the 'consumes' or 'mayConsume' call.

*/
//
// Original Author:  Chris Jones
//         Created:  Wed, 03 Apr 2013 17:54:11 GMT
//

// system include files

// user include files

// forward declarations
namespace edm {
  class EDConsumerBase;
  template <typename T>
  class EDGetTokenT;

  class EDGetToken {
    friend class EDConsumerBase;

  public:
    constexpr EDGetToken() noexcept : m_value{s_uninitializedValue} {}

    template <typename T>
    constexpr EDGetToken(EDGetTokenT<T> iOther) noexcept : m_value{iOther.m_value} {}

    constexpr EDGetToken(const EDGetToken&) noexcept = default;
    constexpr EDGetToken(EDGetToken&&) noexcept = default;
    constexpr EDGetToken& operator=(const EDGetToken&) noexcept = default;
    constexpr EDGetToken& operator=(EDGetToken&&) noexcept = default;

    // ---------- const member functions ---------------------
    constexpr unsigned int index() const noexcept { return m_value; }
    constexpr bool isUninitialized() const noexcept { return m_value == s_uninitializedValue; }

  private:
    //for testing
    friend class TestEDGetToken;

    static const unsigned int s_uninitializedValue = 0xFFFFFFFF;

    constexpr explicit EDGetToken(unsigned int iValue) noexcept : m_value(iValue) {}

    // ---------- member data --------------------------------
    unsigned int m_value;
  };

  template <typename T>
  class EDGetTokenT {
    friend class EDConsumerBase;
    friend class EDGetToken;

  public:
    constexpr EDGetTokenT() : m_value{s_uninitializedValue} {}

    constexpr EDGetTokenT(const EDGetTokenT<T>&) noexcept = default;
    constexpr EDGetTokenT(EDGetTokenT<T>&&) noexcept = default;
    constexpr EDGetTokenT& operator=(const EDGetTokenT<T>&) noexcept = default;
    constexpr EDGetTokenT& operator=(EDGetTokenT<T>&&) noexcept = default;

    template <typename ADAPTER>
    constexpr explicit EDGetTokenT(ADAPTER&& iAdapter) : EDGetTokenT(iAdapter.template consumes<T>()) {}

    template <typename ADAPTER>
    constexpr EDGetTokenT& operator=(ADAPTER&& iAdapter) {
      EDGetTokenT<T> temp(iAdapter.template consumes<T>());
      m_value = temp.m_value;

      return *this;
    }

    //Needed to avoid EDGetTokenT(ADAPTER&&) from being called instead
    // when we can use C++20 concepts we can avoid the problem using a constraint
    constexpr EDGetTokenT(EDGetTokenT<T>& iOther) noexcept : m_value{iOther.m_value} {}
    constexpr EDGetTokenT(const EDGetTokenT<T>&& iOther) noexcept : m_value{iOther.m_value} {}

    constexpr EDGetTokenT& operator=(EDGetTokenT<T>& iOther) {
      return (*this = const_cast<const EDGetTokenT<T>&>(iOther));
    }
    // ---------- const member functions ---------------------
    constexpr unsigned int index() const noexcept { return m_value; }
    constexpr bool isUninitialized() const noexcept { return m_value == s_uninitializedValue; }

  private:
    //for testing
    friend class TestEDGetToken;

    static const unsigned int s_uninitializedValue = 0xFFFFFFFF;

    constexpr explicit EDGetTokenT(unsigned int iValue) noexcept : m_value(iValue) {}

    // ---------- member data --------------------------------
    unsigned int m_value;
  };
}  // namespace edm

#endif
