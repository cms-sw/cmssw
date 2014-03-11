#ifndef FWCore_ServiceRegistry_RandomEngineSentry_h
#define FWCore_ServiceRegistry_RandomEngineSentry_h

/**\class edm::RandomEngineSentry

Description:

*/
//
// Original Author: W. David Dagenhart
//         Created: 11/26/2013

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

namespace CLHEP {
  class HepRandomEngine;
}

namespace edm {

  class LuminosityBlockIndex;
  class StreamID;

  template <class T> class RandomEngineSentry {
  public:

    explicit RandomEngineSentry(T* t, CLHEP::HepRandomEngine* engine): t_(t) {
      if(t) {
        t->setRandomEngine(engine);
      }
    }

    explicit RandomEngineSentry(T* t, StreamID const& streamID): t_(t) {
      if(t) {
        Service<RandomNumberGenerator> rng;
        if (!rng.isAvailable()) {
          throw cms::Exception("Configuration")
              << "Attempt to get a random engine when the RandomNumberGeneratorService is not configured.\n"
                 "You must configure the service if you want an engine.\n";
        }
        CLHEP::HepRandomEngine& engine = rng->getEngine(streamID);
        t->setRandomEngine(&engine);
      }
    }

    explicit RandomEngineSentry(T* t, LuminosityBlockIndex const& lumi): t_(t) {
      if(t) {
        Service<RandomNumberGenerator> rng;
        if (!rng.isAvailable()) {
          throw cms::Exception("Configuration")
              << "Attempt to get a random engine when the RandomNumberGeneratorService is not configured.\n"
                 "You must configure the service if you want an engine.\n";
        }
        CLHEP::HepRandomEngine& engine = rng->getEngine(lumi);
        t->setRandomEngine(&engine);
      }
    }

    ~RandomEngineSentry() { if(t_) t_->setRandomEngine(nullptr); }

  private:
    T* t_;
  };
}
#endif
