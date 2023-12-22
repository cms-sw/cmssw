#ifndef FWCore_Framework_stream_moduleAbilities_h
#define FWCore_Framework_stream_moduleAbilities_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     moduleAbilities
//
/**\file moduleAbilities moduleAbilities.h "FWCore/Framework/interface/one/moduleAbilities.h"

 Description: Template arguments which only apply to stream::{Module} classes

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri, 22 Dec 2023 19:38:53 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/moduleAbilities.h"

// forward declarations

namespace edm {
  namespace stream {
    struct WatchRuns {
      static constexpr module::Abilities kAbilities = module::Abilities::kStreamWatchRuns;
      using Type = module::Empty;
    };

    struct WatchLuminosityBlocks {
      static constexpr module::Abilities kAbilities = module::Abilities::kStreamWatchLuminosityBlocks;
      using Type = module::Empty;
    };
  }  // namespace stream
}  // namespace edm

#endif
