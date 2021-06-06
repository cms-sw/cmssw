#ifndef FWCore_Utilities_ESGetTokenGeneric_h
#define FWCore_Utilities_ESGetTokenGeneric_h
// -*- C++ -*-
//
// Package:     FWCore/Utilities
// Class  :     ESGetTokenGeneric
//
/**\class ESGetTokenGeneric ESGetTokenGeneric.h "FWCore/Utilities/interface/ESGetTokenGeneric.h"

 Description: A token used call EventSetupRecord::doGet

 Usage:
    An ESGetTokenGeneric is created by calls to 'esConsumes' from an ED module.
*/

#include "FWCore/Utilities/interface/ESIndices.h"
#include "FWCore/Utilities/interface/TypeIDBase.h"
#include <limits>

namespace edm {
  class EDConsumerBase;
  class EventSetup;
  class EventSetupImpl;
  namespace eventsetup {
    class EventSetupRecord;
  }

  // A ESGetTokenGeneric is created by calls to 'esConsumes' from an EDM
  // module.
  class ESGetTokenGeneric {
    friend class EDConsumerBase;
    friend class EventSetup;
    friend class EventSetupImpl;
    friend class eventsetup::EventSetupRecord;

  public:
    constexpr ESGetTokenGeneric() noexcept = default;

    constexpr ESGetTokenGeneric(ESGetTokenGeneric const&) noexcept = default;
    constexpr ESGetTokenGeneric(ESGetTokenGeneric&&) noexcept = default;

    constexpr ESGetTokenGeneric& operator=(ESGetTokenGeneric&&) noexcept = default;
    constexpr ESGetTokenGeneric& operator=(ESGetTokenGeneric const&) noexcept = default;

    constexpr unsigned int transitionID() const noexcept { return m_transitionID; }
    constexpr TypeIDBase const& recordType() const noexcept { return m_recordType; }
    constexpr bool isInitialized() const noexcept { return transitionID() != std::numeric_limits<unsigned int>::max(); }
    constexpr ESTokenIndex index() const noexcept { return m_index; }
    constexpr bool hasValidIndex() const noexcept { return index() != invalidIndex(); }
    static constexpr ESTokenIndex invalidIndex() noexcept { return ESTokenIndex{std::numeric_limits<int>::max()}; }

  private:
    explicit constexpr ESGetTokenGeneric(unsigned int transitionID,
                                         ESTokenIndex index,
                                         TypeIDBase const& recordType) noexcept
        : m_recordType{recordType}, m_transitionID{transitionID}, m_index{index} {}

    TypeIDBase m_recordType{};
    unsigned int m_transitionID{std::numeric_limits<unsigned int>::max()};
    ESTokenIndex m_index{std::numeric_limits<int>::max()};
  };

}  // namespace edm

#endif
