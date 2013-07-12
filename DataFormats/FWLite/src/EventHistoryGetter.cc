// -*- C++ -*-
//
// Package:     DataFormats
// Class  :     EventHistoryGetter
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:
//         Created:  Wed Feb 10 11:15:18 CST 2010
// $Id: EventHistoryGetter.cc,v 1.1 2010/02/12 15:26:08 ewv Exp $
//

// user include files
#include "DataFormats/FWLite/interface/EventHistoryGetter.h"


namespace fwlite {

    //
    // constructors and destructor
    //
    EventHistoryGetter::EventHistoryGetter(const Event* event) {
        event_ = event;
    }

    EventHistoryGetter::~EventHistoryGetter() {}

    //
    // const member functions
    //
    const edm::ProcessHistory& EventHistoryGetter::history() const {
        return event_->history();
    }
}
