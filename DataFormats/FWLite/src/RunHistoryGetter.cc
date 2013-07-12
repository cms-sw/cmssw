// -*- C++ -*-
//
// Package:     DataFormats
// Class  :     LumiHistoryGetter
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:
//         Created:  Wed Feb 10 11:15:18 CST 2010
// $Id: RunHistoryGetter.cc,v 1.1 2010/02/11 17:21:40 ewv Exp $
//

// user include files
#include "DataFormats/FWLite/interface/RunHistoryGetter.h"


namespace fwlite {

    //
    // constructors and destructor
    //
    RunHistoryGetter::RunHistoryGetter(const Run* run) {
        run_ = run;
    }

    RunHistoryGetter::~RunHistoryGetter() {}

    //
    // const member functions
    //
    const edm::ProcessHistory& RunHistoryGetter::history() const {
        return run_->history();
    }
}
