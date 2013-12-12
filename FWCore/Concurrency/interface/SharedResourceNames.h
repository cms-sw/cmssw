#ifndef FWCore_Concurrency_SharedResourceNames_h
#define FWCore_Concurrency_SharedResourceNames_h
//
// Package: Concurrency
// Class : ShareResourceNames
//
/**\class edm::SharedResourceNames

Description: Contains the names of external shared resources.

*/
//
// Original Author: W. David Dagenhart
// Created: 19 November 2013
//

#include <string>

namespace edm {
  class SharedResourceNames {
  public:
    // GEANT 4.9.X needs to be declared a shared resource
    // In the future, 4.10.X and later might not need to be
    static const std::string kGEANT;
    static const std::string kCLHEPRandomEngine;
    static const std::string kPythia6;
    static const std::string kPythia8;
    static const std::string kPhotos;
    static const std::string kTauola;
    static const std::string kEvtGen;
  };

  std::string uniqueSharedResourceName();
}
#endif
