#ifndef FWCore_Utilities_ESGetToken_h
#define FWCore_Utilities_ESGetToken_h
// -*- C++ -*-
//
// Package:     FWCore/Utilities
// Class  :     ESGetToken
//
/**\class ESGetToken ESGetToken.h "FWCore/Utilities/interface/ESGetToken.h"

 Description: A token used to get data from the event setup system

 Usage:
    An ESGetToken is created by calls to 'esConsumes' from an ED module
    or via a ConsumesCollector::consumes.
*/

#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/ESIndices.h"
#include <limits>

namespace edm {
  class EDConsumerBase;
  class ESProducer;
  class ESConsumesCollector;
  class EventSetup;
  class EventSetupImpl;
  namespace eventsetup {
    class EventSetupRecord;
  }

  // A ESGetToken is created by calls to 'esConsumes' from an EDM
  // module.
  template <typename ESProduct, typename ESRecord>
  class ESGetToken {
    friend class EDConsumerBase;
    friend class ESProducer;
    friend class ESConsumesCollector;
    friend class EventSetup;
    friend class EventSetupImpl;
    friend class eventsetup::EventSetupRecord;

  public:
    constexpr ESGetToken() noexcept = default;

    constexpr ESGetToken(ESGetToken<ESProduct, ESRecord> const&) noexcept = default;
    constexpr ESGetToken(ESGetToken<ESProduct, ESRecord>&&) noexcept = default;

    constexpr ESGetToken<ESProduct, ESRecord>& operator=(ESGetToken<ESProduct, ESRecord>&&) noexcept = default;
    constexpr ESGetToken<ESProduct, ESRecord>& operator=(ESGetToken<ESProduct, ESRecord> const&) noexcept = default;

    template <typename ADAPTER>
    constexpr explicit ESGetToken(ADAPTER&& iAdapter) : ESGetToken(iAdapter.template consumes<ESProduct, ESRecord>()) {}

    template <typename ADAPTER>
    constexpr ESGetToken<ESProduct, ESRecord>& operator=(ADAPTER&& iAdapter) noexcept {
      ESGetToken<ESProduct, ESRecord> temp(std::forward<ADAPTER>(iAdapter));
      return (*this = std::move(temp));
    }

    //protect against templated version being a better match
    constexpr ESGetToken(ESGetToken<ESProduct, ESRecord>& iOther)
        : ESGetToken(const_cast<const ESGetToken<ESProduct, ESRecord>&>(iOther)) {}
    constexpr ESGetToken<ESProduct, ESRecord>& operator=(ESGetToken<ESProduct, ESRecord>& iOther) noexcept {
      return (*this = const_cast<ESGetToken<ESProduct, ESRecord> const&>(iOther));
    }
    constexpr ESGetToken(ESGetToken<ESProduct, ESRecord> const&& iOther) : ESGetToken(iOther) {}
    constexpr ESGetToken<ESProduct, ESRecord>& operator=(ESGetToken<ESProduct, ESRecord> const&& iOther) noexcept {
      return (*this = iOther);
    }

    constexpr unsigned int transitionID() const noexcept { return m_transitionID; }
    constexpr bool isInitialized() const noexcept { return transitionID() != std::numeric_limits<unsigned int>::max(); }
    constexpr ESTokenIndex index() const noexcept { return m_index; }
    constexpr bool hasValidIndex() const noexcept { return index() != invalidIndex(); }
    static constexpr ESTokenIndex invalidIndex() noexcept { return ESTokenIndex{std::numeric_limits<int>::max()}; }

  private:
    explicit constexpr ESGetToken(unsigned int transitionID, ESTokenIndex index, char const* label) noexcept
        : m_label{label}, m_transitionID{transitionID}, m_index{index} {}

    constexpr char const* name() const noexcept { return m_label; }
    char const* m_label{nullptr};
    unsigned int m_transitionID{std::numeric_limits<unsigned int>::max()};
    ESTokenIndex m_index{std::numeric_limits<int>::max()};
  };

}  // namespace edm

#endif
