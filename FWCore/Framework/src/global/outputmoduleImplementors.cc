// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     outputmoduleImplementors
//
// Implementation:
//     Explicitly instantiate implementor templates for OutputModuleBase
//
//

// system include files

// user include files
#include "FWCore/Framework/interface/global/OutputModule.h"
#include "FWCore/Framework/src/global/implementorsMethods.h"

namespace edm {

class ModuleCallingContext;

namespace global {
namespace outputmodule {
void InputFileWatcher::doRespondToOpenInputFile_(FileBlock const& iB) {
  respondToOpenInputFile(iB);
}
void InputFileWatcher::doRespondToCloseInputFile_(FileBlock const& iB) {
  respondToCloseInputFile(iB);
}
}
}
}
