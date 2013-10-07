// -*- C++ -*-
//
// Package:     Subsystem/Package
// Class  :     SharedResourcesAcquirer
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Sun, 06 Oct 2013 19:43:28 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"


namespace edm {
  void SharedResourcesAcquirer::lock() {
    for(auto m : m_resources) {
      m->lock();
    }
  }
  
  void SharedResourcesAcquirer::unlock() {
    for(auto it = m_resources.rbegin(), itEnd = m_resources.rend();
        it != itEnd; ++it) {
      (*it)->unlock();
    }
  }
}