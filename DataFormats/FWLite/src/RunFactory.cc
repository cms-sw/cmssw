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
// $Id: RunFactory.cc,v 1.1 2010/02/18 20:44:58 ewv Exp $
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

    boost::shared_ptr<fwlite::Run> RunFactory::makeRun(boost::shared_ptr<BranchMapReader> branchMap) const {
        if (not run_) {
            run_ = boost::shared_ptr<fwlite::Run>(new fwlite::Run(branchMap));
        }
        return run_;
    }

}