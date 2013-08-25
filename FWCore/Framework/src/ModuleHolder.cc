// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     ModuleHolder
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Fri, 23 Aug 2013 17:47:05 GMT
//

// system include files

// user include files
#include "FWCore/Framework/src/ModuleHolder.h"
#include "FWCore/Framework/src/WorkerMaker.h"


namespace edm {
  namespace maker {
    std::unique_ptr<Worker>
    ModuleHolder::makeWorker(ExceptionToActionTable const* iActions) const {
      return m_maker->makeWorker(iActions,this);
    }
  }
}