// -*- C++ -*-
//
// Package:     Framework
// Class  :     ValidityInterval
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Tue Mar 29 14:47:31 EST 2005
//

// system include files

// user include files
#include "FWCore/Framework/interface/ValidityInterval.h"
#include <ostream>

namespace edm {
  //
  // constants, enums and typedefs
  //

  //
  // static data member definitions
  //

  //
  // constructors and destructor
  //
  ValidityInterval::ValidityInterval()
      : first_(IOVSyncValue::invalidIOVSyncValue()), last_(IOVSyncValue::invalidIOVSyncValue()) {}

  ValidityInterval::ValidityInterval(const IOVSyncValue& iFirst, const IOVSyncValue& iLast)
      : first_(iFirst), last_(iLast) {}

  // ValidityInterval::ValidityInterval(const ValidityInterval& rhs)
  // {
  //    // do actual copying here;
  // }

  //ValidityInterval::~ValidityInterval()
  //{
  //}

  //
  // assignment operators
  //
  // const ValidityInterval& ValidityInterval::operator=(const ValidityInterval& rhs)
  // {
  //   //An exception safe implementation is
  //   ValidityInterval temp(rhs);
  //   swap(rhs);
  //
  //   return *this;
  // }

  //
  // member functions
  //

  //
  // const member functions
  //
  bool ValidityInterval::validFor(const IOVSyncValue& iInstance) const {
    return first_ <= iInstance && iInstance <= last_;
  }

  //
  // static member functions
  //
  const ValidityInterval& ValidityInterval::invalidInterval() {
    static const ValidityInterval s_invalid;
    return s_invalid;
  }

  std::ostream& operator<<(std::ostream& oStream, ValidityInterval const& iInterval) {
    oStream << "ValidityInterval{" << iInterval.first() << ", " << iInterval.last() << "}";
    return oStream;
  }
}  // namespace edm
