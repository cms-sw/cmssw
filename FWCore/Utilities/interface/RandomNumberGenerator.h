#ifndef FWCore_Utilities_RandomNumberGenerator_h
#define FWCore_Utilities_RandomNumberGenerator_h
// -*- C++ -*-
//
// Package:     Utilities
// Class  :     RandomNumberGenerator
// 
/**\class RandomNumberGenerator RandomNumberGenerator.h FWCore/Utilities/interface/RandomNumberGenerator.h

  Description: Interface for obtaining random number engines and/or seeds.

  Usage:  This class is the abstract interface to a Service which provides access
to the CLHEP random number engines which are used generate random numbers.  One
accesses the service using the Service system.

  edm::Service<RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine& engine = rng->getEngine();

Alternately, one can access seeds that can be used to initialize random
number engines.  This option is mainly provided for backward compatibility
as the getEngine function above did not exist until a lot of code that uses
this mySeed function had already been written.  Note this only returns one
seed.  It has been historically used to initialize the CLHEP::HepJamesRandom
engine which only requires one seed.

  edm::Service<RandomNumberGenerator> rng;
  uint32_t seed = rng->mySeed();

The RandomNumberGenerator automatically knows what module is requesting
an engine or a seed and will return the proper one for that module.

When a separate Producer module is also included in the path the state
of all the engines managed by this service can be saved to the event.
Then in a later process, the RandomNumberGenerator is capable of restoring
the state of the engines from the event in order to be able to exactly
reproduce the earlier process.  (Currently, this restore only works
when random numbers are only generated in modules and not in the source).

*/
//
// Original Author:  Chris Jones, W. David Dagenhart
//         Created:  Tue Mar  7 09:30:28 EST 2006
//

#include <vector>
#include <string>
#include <stdint.h>

namespace CLHEP {
  class HepRandomEngine;
}

namespace edm {

  class Event;

  class RandomNumberGenerator
  {

  public:

    RandomNumberGenerator() {}
    virtual ~RandomNumberGenerator();

    virtual CLHEP::HepRandomEngine& getEngine() const = 0;    

    virtual uint32_t mySeed() const = 0;

    // The following functions should not be used by general users.  They
    // should only be called by code designed to work with the service while
    // it is saving the engine state to an event or restoring it from an event.
    // The first 3 are called by a dedicated producer module (RandomEngineStateProducer).
    // The other two by the InputSource base class.

    virtual const std::vector<std::string>& getCachedLabels() const = 0;
    virtual const std::vector<std::vector<uint32_t> >& getCachedStates() const = 0;
    virtual const std::vector<std::vector<uint32_t> >& getCachedSeeds() const = 0;

    virtual void snapShot() = 0;
    virtual void restoreState(const Event& event) = 0;

    // For debugging purposes only
    virtual void print() = 0;
    virtual void saveEngineState(const std::string& fileName) = 0;
    virtual void restoreEngineState(const std::string& fileName) = 0;

  private:

    RandomNumberGenerator(const RandomNumberGenerator&); // stop default
    const RandomNumberGenerator& operator=(const RandomNumberGenerator&); // stop default
  };
}
#endif
