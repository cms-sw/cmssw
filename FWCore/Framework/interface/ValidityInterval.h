#ifndef Framework_ValidityInterval_h
#define Framework_ValidityInterval_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ValidityInterval
//
/**\class ValidityInterval ValidityInterval.h FWCore/Framework/interface/ValidityInterval.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Tue Mar 29 14:47:25 EST 2005
//

// system include files
#include <iosfwd>
#include <format>

// user include files
#include "FWCore/Framework/interface/IOVSyncValue.h"

// forward declarations
namespace edm {
  class ValidityInterval {
  public:
    ValidityInterval();
    ValidityInterval(const IOVSyncValue& iFirst, const IOVSyncValue& iLast);
    //virtual ~ValidityInterval();

    // ---------- const member functions ---------------------
    bool validFor(const IOVSyncValue&) const;

    const IOVSyncValue& first() const { return first_; }
    const IOVSyncValue& last() const { return last_; }

    bool operator==(const ValidityInterval& iRHS) const { return iRHS.first_ == first_ && iRHS.last_ == last_; }
    bool operator!=(const ValidityInterval& iRHS) const { return !(*this == iRHS); }

    // ---------- static member functions --------------------
    static const ValidityInterval& invalidInterval();

    // ---------- member functions ---------------------------
    void setFirst(const IOVSyncValue& iTime) { first_ = iTime; }
    void setLast(const IOVSyncValue& iTime) { last_ = iTime; }

  private:
    //ValidityInterval(const ValidityInterval&); // stop default

    //const ValidityInterval& operator=(const ValidityInterval&); // stop default

    // ---------- member data --------------------------------
    IOVSyncValue first_;
    IOVSyncValue last_;
  };
  std::ostream& operator<<(std::ostream& oStream, ValidityInterval const& iInterval);

}  // namespace edm

template <>
struct std::formatter<edm::ValidityInterval> : std::formatter<std::string_view> {
  auto format(const edm::ValidityInterval& interval, auto& ctx) {
    std::string temp;
    std::format_to(std::back_inserter(temp), "ValidityInterval{{ {}, {} }}", interval.first(), interval.last());
    return std::formatter<std::string_view>::format(temp, ctx);
  }
};
#endif
