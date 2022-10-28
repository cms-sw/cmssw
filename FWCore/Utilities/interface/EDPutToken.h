#ifndef FWCore_Utilities_EDPutToken_h
#define FWCore_Utilities_EDPutToken_h
// -*- C++ -*-
//
// Package:     FWCore/Utilities
// Class  :     EDPutToken
//
/**\class EDPutToken EDPutToken.h "FWCore/Utilities/interface/EDPutToken.h"

 Description: A Token used to put data into the EDM

 Usage:
    A EDPutToken is created by calls to 'produces'from an EDProducer or EDFilter.
 The EDPutToken can then be used to quickly put data into the edm::Event, edm::LuminosityBlock or edm::Run.
 
The templated form, EDPutTokenT<T>, is the same as EDPutToken except when used to get data the framework
 will skip checking that the type being requested matches the type specified during the 'produces'' call.

*/
//
// Original Author:  Chris Jones
//         Created:  Mon, 18 Sep 2017 17:54:11 GMT
//

// system include files

// user include files

// forward declarations
namespace edm {
  class ProductRegistryHelper;
  template <typename T>
  class EDPutTokenT;
  namespace test {
    class TestProcessorConfig;
  }

  class EDPutToken {
    friend class ProductRegistryHelper;

  public:
    using value_type = unsigned int;

    constexpr EDPutToken() noexcept : m_value{s_uninitializedValue} {}

    template <typename T>
    constexpr EDPutToken(EDPutTokenT<T> iOther) noexcept : m_value{iOther.m_value} {}

    // ---------- const member functions ---------------------
    constexpr value_type index() const noexcept { return m_value; }
    constexpr bool isUninitialized() const noexcept { return m_value == s_uninitializedValue; }

  private:
    //for testing
    friend class TestEDPutToken;

    static constexpr unsigned int s_uninitializedValue = 0xFFFFFFFF;

    explicit constexpr EDPutToken(unsigned int iValue) noexcept : m_value(iValue) {}

    // ---------- member data --------------------------------
    value_type m_value;
  };

  template <typename T>
  class EDPutTokenT {
    friend class ProductRegistryHelper;
    friend class EDPutToken;
    friend class edm::test::TestProcessorConfig;

  public:
    using value_type = EDPutToken::value_type;

    constexpr EDPutTokenT() noexcept : m_value{s_uninitializedValue} {}

    constexpr EDPutTokenT(const EDPutTokenT<T>&) noexcept = default;
    constexpr EDPutTokenT(EDPutTokenT<T>&&) noexcept = default;
    constexpr EDPutTokenT(EDPutTokenT<T>& iToken) noexcept : EDPutTokenT(const_cast<EDPutTokenT<T> const&>(iToken)) {}

    template <typename ADAPTER>
    constexpr explicit EDPutTokenT(ADAPTER&& iAdapter) noexcept : EDPutTokenT(iAdapter.template produces<T>()) {}

    constexpr EDPutTokenT& operator=(const EDPutTokenT<T>&) noexcept = default;
    constexpr EDPutTokenT& operator=(EDPutTokenT<T>&&) noexcept = default;
    constexpr EDPutTokenT& operator=(EDPutTokenT<T>& iOther) noexcept {
      m_value = iOther.m_value;
      return *this;
    }

    template <typename ADAPTER>
    constexpr EDPutTokenT& operator=(ADAPTER&& iAdapter) noexcept {
      EDPutTokenT<T> temp(iAdapter.template produces<T>());
      m_value = temp.m_value;

      return *this;
    }

    // ---------- const member functions ---------------------
    constexpr value_type index() const noexcept { return m_value; }
    constexpr bool isUninitialized() const noexcept { return m_value == s_uninitializedValue; }

  private:
    //for testing
    friend class TestEDPutToken;

    static constexpr unsigned int s_uninitializedValue = 0xFFFFFFFF;

    constexpr explicit EDPutTokenT(unsigned int iValue) noexcept : m_value(iValue) {}
    constexpr explicit EDPutTokenT(unsigned long int iValue) noexcept : m_value(iValue) {}

    // ---------- member data --------------------------------
    value_type m_value;
  };
}  // namespace edm

#endif
