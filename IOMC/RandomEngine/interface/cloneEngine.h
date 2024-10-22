#ifndef IOMC_RandomEngine_cloneEngine_h
#define IOMC_RandomEngine_cloneEngine_h
// -*- C++ -*-
//
// Package:     IOMC/RandomEngine
// Class  :     cloneEngine
//
/**\function cloneEngine cloneEngine.h "IOMC/RandomEngine/interface/cloneEngine.h"

 Description: Function used to clone a random number engine

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Fri, 02 Dec 2022 19:32:10 GMT
//
#include <memory>

namespace CLHEP {
  class HepRandomEngine;
}

namespace edm {
  std::unique_ptr<CLHEP::HepRandomEngine> cloneEngine(CLHEP::HepRandomEngine const&);
};

#endif
