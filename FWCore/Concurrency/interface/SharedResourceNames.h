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

  //ES modules can not share resources with ED modules
  class ESSharedResourceNames {
  public:
    static const std::string kDDGeometry;
    static const std::string kDD4hep;
  };

  // Each time the following function is called, it returns a different
  // name.  The purpose is to address the following problem.
  // A few classes that are modules sometimes use shared resources
  // and sometimes do not use shared resources. It depends on their
  // configuration and also on their template parameters. These classes
  // have to always inherit from the SharedResources base class. If
  // they do not declare any share resources the Framework assumes
  // they depend on all possible shared resources. This causes performance
  // problems. In the cases where they really do not use any shared
  // resources, one has to declare something to avoid the default
  // assumption that they depend on everything. If a nonexistent
  // shared resource is declared and as long as nothing else declares
  // the same shared resource name, there will be no performance effects.
  // This function provides a unique name to be used for that purpose.
  std::string uniqueSharedResourceName();
}  // namespace edm
#endif
