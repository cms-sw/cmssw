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
#include "FWCore/Utilities/interface/get_underlying_safe.h"

namespace CLHEP {
  class HepRandomEngine;
}

namespace edm {

  class LuminosityBlockIndex;
  class StreamID;

  template <class T>
  class RandomEngineSentry {
  public:
    explicit RandomEngineSentry(T* t, CLHEP::HepRandomEngine* engine) : t_(t), engine_(engine) {
      if (t) {
        t->setRandomEngine(engine);
      }
    }

    explicit RandomEngineSentry(T* t, StreamID const& streamID) : t_(t), engine_(nullptr) {
      if (t) {
        Service<RandomNumberGenerator> rng;
        if (!rng.isAvailable()) {
          throw cms::Exception("Configuration")
              << "Attempt to get a random engine when the RandomNumberGeneratorService is not configured.\n"
                 "You must configure the service if you want an engine.\n";
        }
        engine_ = &rng->getEngine(streamID);
        t->setRandomEngine(engine_);
      }
    }

    explicit RandomEngineSentry(T* t, LuminosityBlockIndex const& lumi) : t_(t), engine_(nullptr) {
      if (t) {
        Service<RandomNumberGenerator> rng;
        if (!rng.isAvailable()) {
          throw cms::Exception("Configuration")
              << "Attempt to get a random engine when the RandomNumberGeneratorService is not configured.\n"
                 "You must configure the service if you want an engine.\n";
        }
        engine_ = &rng->getEngine(lumi);
        t->setRandomEngine(engine_);
      }
    }

    ~RandomEngineSentry() {
      if (t_)
        t_->setRandomEngine(nullptr);
    }

    CLHEP::HepRandomEngine const* randomEngine() const { return get_underlying_safe(engine_); }
    CLHEP::HepRandomEngine*& randomEngine() { return get_underlying_safe(engine_); }

  private:
    edm::propagate_const<T*> t_;
    edm::propagate_const<CLHEP::HepRandomEngine*> engine_;
  };
}  // namespace edm
#endif
