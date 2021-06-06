// -*- C++ -*-
//
// Package:     DataFormats/Common
// Class  :     ELseverityLevel
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Wed, 20 Jan 2021 15:39:12 GMT
//

// system include files
#include <cassert>
#include <array>

// user include files
#include "DataFormats/Common/interface/ELseverityLevel.h"

namespace edm {

  std::string_view ELseverityLevel::getName() const noexcept {
    static const auto names = []() {
      std::array<const char*, nLevels> ret;
      ret[ELsev_noValueAssigned] = "?no value?";
      ret[ELsev_zeroSeverity] = "--";
      ret[ELsev_success] = "Debug";
      ret[ELsev_info] = "Info";
      ret[ELsev_fwkInfo] = "FwkInfo";
      ret[ELsev_warning] = "Warning";
      ret[ELsev_error] = "Error";
      ret[ELsev_unspecified] = "??";
      ret[ELsev_severe] = "System";
      ret[ELsev_highestSeverity] = "!!";
      return ret;
    }();

    assert(myLevel < nLevels);
    return names[myLevel];
  }
}  // namespace edm
