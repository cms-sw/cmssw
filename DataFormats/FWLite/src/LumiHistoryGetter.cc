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
//

// user include files
#include "DataFormats/FWLite/interface/LumiHistoryGetter.h"

namespace fwlite {

  //
  // constructors and destructor
  //
  LumiHistoryGetter::LumiHistoryGetter(const LuminosityBlock* lumi) { lumi_ = lumi; }

  LumiHistoryGetter::~LumiHistoryGetter() {}

  //
  // const member functions
  //
  const edm::ProcessHistory& LumiHistoryGetter::history() const { return lumi_->history(); }
}  // namespace fwlite
