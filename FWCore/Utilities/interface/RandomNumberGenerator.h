#ifndef FWCore_Utilities_RandomNumberGenerator_h
#define FWCore_Utilities_RandomNumberGenerator_h

/** \class edm::RandomNumberGenerator

  Description: Interface for obtaining random number engines.

  Usage:  This class is the abstract interface to a Service
which provides access to the random number engines which are
used to generate random numbers. One accesses the service using
the Service system.

  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine& engine = rng->getEngine(streamID);
  or
  CLHEP::HepRandomEngine& engine = rng->getEngine(luminosityBlockIndex);

The RandomNumberGenerator automatically knows what module is
requesting an engine and will return the proper one for that
module.

For each module the service will hold one engine per stream.
In addition, for each module the service will hold a number
of engines equal to the number of LuminosityBlocks that can
be processed concurrently. When running a multithreaded
process the correct engine for the current module and
streamID/luminosityBlockIndex must be used to avoid
data races. This is also important for replay.

A source cannot use this service and sources should not generate
random numbers (otherwise replay will fail).

Random numbers should only be generated in two functions of a
module, the function used to process the Event and also the
global beginLuminosityBlock function. Random numbers should
not be generated at any other time, not in the constructor,
not at beginJob, not at beginRun ... Note that this restriction
applies to generating the random numbers. If one is only calling
the function getEngine that takes a streamID argument then it is
also OK to call it in the beginStream method, but only to get
the reference to the engine and save it or save a pointer to
it (but not to generate any random numbers in beginStream).

The service owns the engines and handles memory management
for them.

The service does a lot of work behind the scenes to allow one
to replay specific events of a prior process.  There are two
different mechanisms.

First, if the parameter named "saveFileName" is set the
state of the random engines will be written to a text file
before each event is processed.  This text file is overwritten
at each event. If a job crashes while processing an event,
then one can replay the processing of the event where the
crash occurred and get the same random number sequences.
For jobs with more than 1 thread or forked processes, the
text files are named by appending "_" and a number
to the "saveFileName" parameter, where the number is either
the child index of the forked process or the streamID of
the stream which processed the event.

Second, if the RandomEngineStateProducer module is executed
the state of all the engines managed by this service
can be saved to both the Event and LuminosityBlock. Then in a
later process, the RandomNumberGenerator is capable of restoring
the state of the engines in order to be able to exactly replay
the earlier process starting at any event without having to
replay the entire process.

This service performs its tasks so that the random sequences
are independent sequences with different seeds for two cases:

1.  Multiprocess jobs where processes are forked
2.  Multithread jobs where multiple threads are used

It is assumed that we will never run jobs that are both
multiprocess and multithread. The service seeding algorithm
will fail to produce independent sequences if that is
attempted.

Three warnings.

1. When there is more than one LuminosityBlock in a single
process, the random number engines used in global begin
luminosity block are reset to the same starting state
before each call to that function (except in replay mode).
This allows the initialization performed in all of them
to be identical. Don't expect unique sequences in different
calls to beginLuminosityBlock.

2. In multiprocess jobs, the engines are reinitialized
after the processes are forked with new seeds equal to
the original seed plus the child index. In multithread
jobs, the stream sequences are initialized in a similar way
by adding the streamID to the seed to form a new
seed for each stream.  The seeds for the engines for
the luminosity blocks are all the same and the original
seed plus the number of streams or the number of forked
child processes is used. In existing work management schemes
this works well, because the initial configured seed is
derived randomly and the seeds in different jobs should
not be close to one another. If not using one of these
work management schemes, one has to be careful to not
configure seeds in multiple jobs that are close enough
together to overlap (closer than the number of forked
processes or streams). For example, if one generated
the original configured seed for a job using the previous
jobs seed and adding one, then there would be a problem.

3. This service properly handles modules running concurrently
and generating random numbers, but not concurrent submodule tasks.
If a module creates its own subtasks that are run concurrently
and generate numbers, then using the engines from the service
will result in data races. If this design if ever needed,
one possible approach would be for each submodule task that
needs random numbers to create its own engine and seed it
using a random number from the module random engine. There
might be better ways.

There are more details explaining this service on a TWIKI page
which can be accessed through a link on the Framework TWIKI page
of the SWGuide.

\author Chris Jones and W. David Dagenhart, created March 7, 2006
*/

#include <cstdint>
#include <iosfwd>
#include <memory>
#include <vector>

class RandomEngineState;

namespace CLHEP {
  class HepRandomEngine;
}

namespace edm {

  class ConsumesCollector;
  class Event;
  class LuminosityBlock;
  class LuminosityBlockIndex;
  class StreamID;

  class RandomNumberGenerator {
  public:
    RandomNumberGenerator() {}
    virtual ~RandomNumberGenerator();

    /// Use the next 2 functions to get the random number engine.
    /// These are the only functions most modules should call.

    /// Use this engine in event methods
    virtual CLHEP::HepRandomEngine& getEngine(StreamID const&) = 0;

    /// Use this engine in the global begin luminosity block method
    virtual CLHEP::HepRandomEngine& getEngine(LuminosityBlockIndex const&) = 0;

    /// This function is not intended for general use. It is intended for
    /// the special case where multiple instances of Pythia 8 will
    /// be run concurrently and we want them to be initialized exactly
    /// the same. In this special case, the luminosity block engine(s)
    /// owned by the service should not be used to generate random numbers
    /// in between calls to cloneEngine, because the clone will be in
    /// the state that existed at the moment of cloning.
    /// Before initializing Pythia, this function should be used to clone
    /// the engine owned by the service and the cloned random engine should be
    /// used to generate numbers for initialization so that all initializations
    /// in the process get identical sequences of random numbers.
    virtual std::unique_ptr<CLHEP::HepRandomEngine> cloneEngine(LuminosityBlockIndex const&) = 0;

    /// This returns the seed from the configuration. In the unusual case where an
    /// an engine type takes multiple seeds to initialize a sequence, this function
    /// only returns the first. As a general rule, this function should not be used,
    /// but is available for backward compatibility and debugging. It might be useful
    /// for some types of tests. Using this to seed engines constructed in modules is
    /// not recommended because (unless done very carefully) it will create duplicate
    /// sequences in different threads and/or data races. Also, if engines are created
    /// by modules the replay mechanism will be broken.
    /// Because it is dangerous and could be misused, this function might be deleted
    /// someday if we ever find time to delete all uses of it in CMSSW. There are of
    /// order 10 last time I checked ...
    virtual std::uint32_t mySeed() const = 0;

    // The following functions should not be used by general users.  They
    // should only be called by Framework code designed to work with the
    // service while it is saving the engine states or restoring them.
    // The first two are called by the EventProcessor at special times.
    // The next two are called by a dedicated producer module (RandomEngineStateProducer).

    virtual void preBeginLumi(LuminosityBlock const& lumi) = 0;
    virtual void postEventRead(Event const& event) = 0;

    virtual void setLumiCache(LuminosityBlockIndex, std::vector<RandomEngineState> const& iStates) = 0;
    virtual void setEventCache(StreamID, std::vector<RandomEngineState> const& iStates) = 0;

    virtual std::vector<RandomEngineState> const& getEventCache(StreamID const&) const = 0;
    virtual std::vector<RandomEngineState> const& getLumiCache(LuminosityBlockIndex const&) const = 0;

    virtual void consumes(ConsumesCollector&& iC) const = 0;

    /// For debugging purposes only.
    virtual void print(std::ostream& os) const = 0;

  private:
    RandomNumberGenerator(RandomNumberGenerator const&) = delete;
    RandomNumberGenerator const& operator=(RandomNumberGenerator const&) = delete;
  };
}  // namespace edm
#endif
