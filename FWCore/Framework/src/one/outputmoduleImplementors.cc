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
// $Id$
//

// system include files

// user include files
#include "FWCore/Framework/src/one/implementorsMethods.h"
#include "FWCore/Framework/interface/one/OutputModule.h"

namespace edm {
  namespace one {
    namespace outputmodule {
      void RunWatcher::doBeginRun_(RunPrincipal const& rp) {
        beginRun(rp);
      }
      void RunWatcher::doEndRun_(RunPrincipal const& rp) {
        endRun(rp);
      }
      
      void LuminosityBlockWatcher::doBeginLuminosityBlock_(LuminosityBlockPrincipal const& lbp) {
        beginLuminosityBlock(lbp);
      }
      void LuminosityBlockWatcher::doEndLuminosityBlock_(LuminosityBlockPrincipal const& lbp) {
        endLuminosityBlock(lbp);
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
