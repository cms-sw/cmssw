#ifndef FWCore_Utilities_RandomNumberGenerator_h
#define FWCore_Utilities_RandomNumberGenerator_h

/** \class edm::RandomNumberGenerator

  Description: Interface for obtaining random number engines.

  Usage:  This class is the abstract interface to a Service
which provides access to the random number engines which are
used generate random numbers. One accesses the service using
the Service system.

  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine& engine = rng->getEngine();

The RandomNumberGenerator automatically knows what module is
requesting an engine and will return the proper one for that
module.

A source cannot use this service and sources should not generate
random numbers.

Random numbers should only be generated in two functions of a
module, the function used to process the Event and also the
beginLuminosityBlock function. Random numbers should not be
generated at any other time, not in the constructor, not at
beginJob, not at beginRun ... Note that this restriction applies
to generating the random numbers only, it is fine to get the
reference to the engine and save it or a pointer to it at other
times as long as it is not used to generate random numbers.

The service owns the engines and handles memory management
for them.

The service does a lot of work behind the scene to allow one
to replay specific events of a prior process.  There are two
different mechanisms.

First, if the parameter named "saveFileName" is set the state
of the random engines will be written to a separate text file
before each event is processed.  This text file is overwritten
at each event. If a job crashes while processing an event,
then one can replay the processing of the event where the crash
occurred and get the same random number sequences.

Second, when a separate Producer module is also included in a
path the state of all the engines managed by this service can
be saved to the both the Event and LuminosityBlock. Then in a
later process, the RandomNumberGenerator is capable of restoring
the state of the engines in order to be able to exactly replay
the earlier process starting at any event without having to replay
the entire process.

This service performs tasks so that the random sequences in
multiprocess jobs are independent sequences with different seeds.

Two warnings.

1. When there is more than one LuminosityBlock in a
single process, the random number engines are reset to
the same starting state at beginLuminosityBlock for all
of them. This allows the initialization performed in all
of them to be identical. At the end of beginLuminosityBlock
(after the first one), the engine states are reset to what
they were before beginLuminosityBlock so that the sequences
for events continue forward to produce independent events.
In current CMS use cases this is the correct behavior,
but one could imagine cases where it is not. This is the
best scheme we could come up with to allow replay in an
environment including random numbers being generated
during initialization and file merging and multi process
jobs.

2. In multiprocess jobs, the sequences for the child processes
are reinitialized with new seeds after the first beginLuminosityBlock
but before any events are processed. The seed used is the
original seed plus the child index. In existing work management
schemes this works well, but if one were to run multiple
overlapping jobs and increment the seeds in the configuration
by 1 for each succeeding job and use multiprocess
jobs, then there would be a problem with the same
seeds being used in the child processes spawned by
different parent processes.

There are more details explaining this service on a TWIKI page
which can be accessed through a link on the Framework TWIKI page
of the SWGuide.

\author Chris Jones and W. David Dagenhart, created March 7, 2006
*/

#include <vector>
#include <stdint.h>

class RandomEngineState;

namespace CLHEP {
  class HepRandomEngine;
}

namespace edm {

  class StreamID;
  class LuminosityBlock;
  class LuminosityBlockIndex;
  class Event;

  class RandomNumberGenerator
  {
  public:

    RandomNumberGenerator() {}
    virtual ~RandomNumberGenerator();

    virtual CLHEP::HepRandomEngine& getEngine(StreamID const&) const { return getEngine(); }
    virtual CLHEP::HepRandomEngine& getEngine(LuminosityBlockIndex const&) const { return getEngine(); }

    /// Exists for backward compatibility.
    virtual uint32_t mySeed() const = 0;

    // The following functions should not be used by general users.  They
    // should only be called by Framework code designed to work with the
    // service while it is saving the engine states or restoring them.
    // The first two are called by the InputSource base class.
    // The next two are called by a dedicated producer module (RandomEngineStateProducer).

    virtual void preBeginLumi(LuminosityBlock const& lumi) = 0;
    virtual void postEventRead(Event const& event) = 0;

    virtual std::vector<RandomEngineState> const& getLumiCache() const = 0;
    virtual std::vector<RandomEngineState> const& getEventCache() const = 0;
 
    /// For debugging purposes only.
    virtual void print() = 0;

  private:

    RandomNumberGenerator(RandomNumberGenerator const&); // stop default
    RandomNumberGenerator const& operator=(RandomNumberGenerator const&); // stop default

    /// Use this to get the random number engine, this is the only function most users should call.
    virtual CLHEP::HepRandomEngine& getEngine() const = 0;
  };
}
#endif
