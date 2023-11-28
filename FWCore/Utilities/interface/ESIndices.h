#ifndef FWCore_Utilities_ESIndices_h
#define FWCore_Utilities_ESIndices_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ESIndices
//
/**\class ESResolverIndex ESIndices.h FWCore/Utilities/interface/ESIndices.h
   \class ESTokenIndex ESIndices.h FWCore/Utilities/interface/ESIndices.h
   \class ESRecordIndex ESIndices.h FWCore/Utilities/interface/ESIndices.h

 Description: Classes representing indices used in the EventSetup System

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Sat Apr  6 14:47:35 EST 2019
//

// system include files
#include <limits>
#include <ostream>

// user include files

namespace edm {
  class ESResolverIndex {
  public:
    using Value_t = int;
    constexpr ESResolverIndex() noexcept = default;
    constexpr explicit ESResolverIndex(Value_t iValue) noexcept : index_{iValue} {}
    constexpr ESResolverIndex(ESResolverIndex const&) noexcept = default;
    constexpr ESResolverIndex(ESResolverIndex&&) noexcept = default;

    constexpr ESResolverIndex& operator=(ESResolverIndex const&) noexcept = default;
    constexpr ESResolverIndex& operator=(ESResolverIndex&&) noexcept = default;

    constexpr bool operator==(ESResolverIndex iOther) const noexcept { return iOther.index_ == index_; }
    constexpr bool operator!=(ESResolverIndex iOther) const noexcept { return iOther.index_ != index_; }

    constexpr Value_t value() const noexcept { return index_; }

  private:
    Value_t index_ = std::numeric_limits<int>::max();
  };

  inline std::ostream& operator<<(std::ostream& iOS, ESResolverIndex const& iIndex) {
    iOS << iIndex.value();
    return iOS;
  }

  class ESTokenIndex {
  public:
    using Value_t = int;

    constexpr ESTokenIndex() noexcept = default;
    constexpr explicit ESTokenIndex(Value_t iValue) noexcept : index_{iValue} {}
    constexpr ESTokenIndex(ESTokenIndex const&) noexcept = default;
    constexpr ESTokenIndex(ESTokenIndex&&) noexcept = default;

    constexpr ESTokenIndex& operator=(ESTokenIndex const&) noexcept = default;
    constexpr ESTokenIndex& operator=(ESTokenIndex&&) noexcept = default;

    constexpr bool operator==(ESTokenIndex iOther) const noexcept { return iOther.index_ == index_; }
    constexpr bool operator!=(ESTokenIndex iOther) const noexcept { return iOther.index_ != index_; }

    constexpr Value_t value() const noexcept { return index_; }

  private:
    Value_t index_ = std::numeric_limits<Value_t>::max();
  };
  inline std::ostream& operator<<(std::ostream& iOS, ESTokenIndex const& iIndex) {
    iOS << iIndex.value();
    return iOS;
  }

  class ESRecordIndex {
  public:
    using Value_t = unsigned int;

    constexpr ESRecordIndex() noexcept = default;
    constexpr explicit ESRecordIndex(unsigned int iValue) noexcept : index_{iValue} {}
    constexpr ESRecordIndex(ESRecordIndex const&) noexcept = default;
    constexpr ESRecordIndex(ESRecordIndex&&) noexcept = default;

    constexpr ESRecordIndex& operator=(ESRecordIndex const&) noexcept = default;
    constexpr ESRecordIndex& operator=(ESRecordIndex&&) noexcept = default;

    constexpr bool operator==(ESRecordIndex iOther) const noexcept { return iOther.index_ == index_; }
    constexpr bool operator!=(ESRecordIndex iOther) const noexcept { return iOther.index_ != index_; }

    constexpr Value_t value() const noexcept { return index_; }

    static constexpr Value_t invalidValue() { return std::numeric_limits<Value_t>::max(); }

  private:
    Value_t index_ = std::numeric_limits<Value_t>::max();
  };
  inline std::ostream& operator<<(std::ostream& iOS, ESRecordIndex const& iIndex) {
    iOS << iIndex.value();
    return iOS;
  }

}  // namespace edm
#endif
