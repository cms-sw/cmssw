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
    namespace impl {
      template class SharedResourcesUser<edm::one::OutputModuleBase>;
    }
    namespace outputmodule {
      void RunWatcher::doBeginRun_(RunForOutput const& r) {
        beginRun(r);
      }
      void RunWatcher::doEndRun_(RunForOutput const& r) {
        endRun(r);
      }
      
      void LuminosityBlockWatcher::doBeginLuminosityBlock_(LuminosityBlockForOutput const& lb) {
        beginLuminosityBlock(lb);
      }
      void LuminosityBlockWatcher::doEndLuminosityBlock_(LuminosityBlockForOutput const& lb) {
        endLuminosityBlock(lb);
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
