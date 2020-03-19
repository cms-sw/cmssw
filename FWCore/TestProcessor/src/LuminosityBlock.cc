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

// system include files

// user include files
#include "FWCore/TestProcessor/interface/LuminosityBlock.h"

namespace edm {
  namespace test {

    //
    // constants, enums and typedefs
    //

    //
    // static data member definitions
    //

    //
    // constructors and destructor
    //
    LuminosityBlock::LuminosityBlock(std::shared_ptr<LuminosityBlockPrincipal const> iPrincipal,
                                     std::string iModuleLabel,
                                     std::string iProcessName)
        : principal_{std::move(iPrincipal)}, label_{std::move(iModuleLabel)}, processName_{std::move(iProcessName)} {}

    //
    // member functions
    //

    //
    // const member functions
    //

    //
    // static member functions
    //

  }  // namespace test
}  // namespace edm
