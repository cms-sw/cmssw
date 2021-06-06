// -*- C++ -*-
//
// Package:     FWCore/TestProcessor
// Class  :     ProcessBlock
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  W. David Dagenhart
//         Created:  28 May 2020
//

#include "FWCore/TestProcessor/interface/ProcessBlock.h"

namespace edm {
  namespace test {

    ProcessBlock::ProcessBlock(ProcessBlockPrincipal const* iPrincipal,
                               std::string iModuleLabel,
                               std::string iProcessName)
        : principal_{iPrincipal}, label_{std::move(iModuleLabel)}, processName_{std::move(iProcessName)} {}

  }  // namespace test
}  // namespace edm
