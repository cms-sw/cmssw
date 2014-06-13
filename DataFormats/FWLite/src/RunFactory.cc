// -*- C++ -*-
//
// Package:     DataFormats/FWLite
// Class  :     RunFactory
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:
//         Created:  Wed Feb 10 11:15:18 CST 2010
//

// system include files

// user include files
#include "DataFormats/FWLite/interface/RunFactory.h"

namespace fwlite {

    //
    // constructors and destructor
    //
    RunFactory::RunFactory() {}
    RunFactory::~RunFactory() {}

    std::shared_ptr<fwlite::Run> RunFactory::makeRun(std::shared_ptr<BranchMapReader> branchMap) const {
        if (not run_) {
            run_ = std::shared_ptr<fwlite::Run>(new fwlite::Run(branchMap));
        }
        return run_;
    }

}
