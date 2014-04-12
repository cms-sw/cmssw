// -*- C++ -*-
//
// Package:     Framework
// Class  :     UnscheduledHandler
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Mon Oct 13 13:58:07 CEST 2008
//

// system include files
#include <cassert>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/UnscheduledHandler.h"

namespace edm {

  UnscheduledHandler::~UnscheduledHandler() {
  }

  bool
  UnscheduledHandler::tryToFill(std::string const& label,
                                EventPrincipal& iEvent,
                                ModuleCallingContext const* mcc) {
     assert(m_setup);
     return tryToFillImpl(label, iEvent, *m_setup, mcc);
  }
}
