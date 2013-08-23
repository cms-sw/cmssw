// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     outputmoduleImplementors
// 
// Implementation:
//     Explicitly instantiate implementor templates for OutputModuleBase
//
// Original Author:  Chris Jones
//         Created:  Thu, 09 May 2013 20:14:06 GMT
//

// system include files

// user include files
#include "FWCore/Framework/src/one/implementorsMethods.h"
#include "FWCore/Framework/interface/one/OutputModule.h"

namespace edm {

  class ModuleCallingContext;

  namespace one {
    namespace outputmodule {
      void RunWatcher::doBeginRun_(RunPrincipal const& rp, ModuleCallingContext const* mcc) {
        beginRun(rp, mcc);
      }
      void RunWatcher::doEndRun_(RunPrincipal const& rp, ModuleCallingContext const* mcc) {
        endRun(rp, mcc);
      }
      
      void LuminosityBlockWatcher::doBeginLuminosityBlock_(LuminosityBlockPrincipal const& lbp, ModuleCallingContext const* mcc) {
        beginLuminosityBlock(lbp, mcc);
      }
      void LuminosityBlockWatcher::doEndLuminosityBlock_(LuminosityBlockPrincipal const& lbp, ModuleCallingContext const* mcc) {
        endLuminosityBlock(lbp, mcc);
      }
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
