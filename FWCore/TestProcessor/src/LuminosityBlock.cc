// -*- C++ -*-
//
// Package:     FWCore/TestProcessor
// Class  :     LuminosityBlock
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Mon, 30 Apr 2018 18:51:33 GMT
//

#include "FWCore/TestProcessor/interface/LuminosityBlock.h"

namespace edm {
  namespace test {

    LuminosityBlock::LuminosityBlock(std::shared_ptr<LuminosityBlockPrincipal const> iPrincipal,
                                     std::string iModuleLabel,
                                     std::string iProcessName)
        : principal_{std::move(iPrincipal)}, label_{std::move(iModuleLabel)}, processName_{std::move(iProcessName)} {}

  }  // namespace test
}  // namespace edm
