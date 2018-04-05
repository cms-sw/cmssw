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
#include "FWCore/Framework/src/limited/implementorsMethods.h"
#include "FWCore/Framework/interface/limited/OutputModule.h"

namespace edm {

  class ModuleCallingContext;

  namespace limited {
    namespace outputmodule {
      void InputFileWatcher::doRespondToOpenInputFile_(FileBlock const& iB)
      {
        respondToOpenInputFile(iB);
      }
      void InputFileWatcher::doRespondToCloseInputFile_(FileBlock const& iB)
      {
        respondToCloseInputFile(iB);
      }
    }
  }
}
