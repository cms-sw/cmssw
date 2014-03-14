#ifndef IOMC_RandomEngine_RandomNumberGeneratorService_h
#define IOMC_RandomEngine_RandomNumberGeneratorService_h

/** \class edm::service::RandomNumberGeneratorService

 Description: Manages random number engines for modules

 Usage: See comments in base class, FWCore/Utilities/RandomNumberGenerator.h

\author Chris Jones and W. David Dagenhart, created March 7, 2006
  (originally in FWCore/Services)
*/

#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "boost/shared_ptr.hpp"

#include <fstream>
#include <map>
#include <stdint.h>
#include <string>
#include <vector>

class RandomEngineState;

namespace CLHEP {
  class HepRandomEngine;
}

namespace edm {
  class ActivityRegistry;
  class ConfigurationDescriptions;
  class Event;
  class EventSetup;
  class LuminosityBlock;
  class ModuleDescription;
  class ParameterSet;

  namespace service {

    class RandomNumberGeneratorService : public RandomNumberGenerator {

    public:
      static size_t const maxSeeds = 65536U;

      static size_t const maxStates = 65536U;

      RandomNumberGeneratorService(ParameterSet const& pset, ActivityRegistry& activityRegistry);
      virtual ~RandomNumberGeneratorService();

      /// Exists for backward compatibility.
      virtual uint32_t mySeed() const;

      static void fillDescriptions(ConfigurationDescriptions& descriptions);

      // The following functions should not be used by general users.  They
      // should only be called by code designed to work with the service while
      // it is saving the engine state to an event or restoring it from an event
      // and also used to keep track of which module is currently active.
      // The first 20 are called either by the InputSource base class
      // or via the ActivityRegistry.  The next 2 are called by a dedicated
      // producer module (RandomEngineStateProducer).

      void postForkReacquireResources(unsigned childIndex, unsigned kMaxChildren);

      virtual void preBeginLumi(LuminosityBlock const& lumi);
      void postBeginLumi(LuminosityBlock const& lumi, EventSetup const& es);
      virtual void postEventRead(Event const& event);

      void preModuleConstruction(ModuleDescription const& description);
      void postModuleConstruction(ModuleDescription const& description);

      void preModuleBeginJob(ModuleDescription const& description);
      void postModuleBeginJob(ModuleDescription const& description);

      void preModuleBeginRun(ModuleDescription const& description);
      void postModuleBeginRun(ModuleDescription const& description);

      void preModuleBeginLumi(ModuleDescription const& description);
      void postModuleBeginLumi(ModuleDescription const& description);

      void preModule(ModuleDescription const& description);
      void postModule(ModuleDescription const& description);

      void preModuleEndLumi(ModuleDescription const& description);
      void postModuleEndLumi(ModuleDescription const& description);

      void preModuleEndRun(ModuleDescription const& description);
      void postModuleEndRun(ModuleDescription const& description);

      void preModuleEndJob(ModuleDescription const& description);
      void postModuleEndJob(ModuleDescription const& description);

      virtual std::vector<RandomEngineState> const& getLumiCache() const;
      virtual std::vector<RandomEngineState> const& getEventCache() const;

      /// For debugging purposes only
      virtual void print();

    private:

      typedef std::vector<std::string> VString;
      typedef std::vector<uint32_t> VUint32;

      RandomNumberGeneratorService(RandomNumberGeneratorService const&); // disallow default

      RandomNumberGeneratorService const& operator=(RandomNumberGeneratorService const&); // disallow default

      /// Use this to get the random number engine, this is the only function most users should call.
      virtual CLHEP::HepRandomEngine& getEngine() const;

      // These two functions are called internally to keep track
      // of which module is currently active

      void push(std::string const& iLabel);
      void pop();

      void readFromLuminosityBlock(LuminosityBlock const& lumi);
      void readFromEvent(Event const& event);
      bool backwardCompatibilityRead(Event const& event);

      void snapShot(std::vector<RandomEngineState>& cache);
      void restoreFromCache(std::vector<RandomEngineState> const& cache);

      void checkEngineType(std::string const& typeFromConfig,
                           std::string const& typeFromEvent,
                           std::string const& engineLabel);

      void saveStatesToFile(std::string const& fileName);
      void writeStates(std::vector<RandomEngineState> const& v,
                       std::ofstream& outFile);
      void writeVector(VUint32 const& v,
                       std::ofstream& outFile);
      std::string constructSaveFileName();

      void readEventStatesFromTextFile(std::string const& fileName);
      void readLumiStatesFromTextFile(std::string const& fileName);
      void readStatesFromFile(std::string const& fileName,
                              std::vector<RandomEngineState>& cache,
                              std::string const& whichStates);
      bool readEngineState(std::istream& is,
                           std::vector<RandomEngineState>& cache,
                           std::string const& whichStates,
                           bool& saveToCache);
      void readVector(std::istream& is, unsigned numItems, std::vector<uint32_t>& v);

      void startNewSequencesForEvents();

      // ---------- member data --------------------------------

      // We store the engines using the corresponding module label
      // as a key into a map
      typedef std::map<std::string, boost::shared_ptr<CLHEP::HepRandomEngine> > EngineMap;
      EngineMap engineMap_;

      // The next four help to keep track of the currently active
      // module (its label and associated engine)

      std::vector<EngineMap::const_iterator> engineStack_;
      EngineMap::const_iterator currentEngine_;

      VString labelStack_;
      std::string currentLabel_;

      // This is used for beginRun, endRun, endLumi, beginJob, endJob
      // and constructors in the check that prevents random numbers
      // from being thrown in those methods.
      std::vector<std::vector<unsigned long> > engineStateStack_;

      // These hold the input tags needed to retrieve the states
      // of the random number engines stored in a previous process.
      // If the label in the tag is the empty string (the default),
      // then the service does not try to restore the random numbers.
      edm::InputTag restoreStateTag_;
      edm::InputTag restoreStateBeginLumiTag_;

      std::vector<RandomEngineState> lumiCache_;
      std::vector<RandomEngineState> eventCache_;

      // Keeps track of the seeds used to initialize the engines.
      // Also uses the module label as a key
      std::map<std::string, VUint32> seedMap_;
      std::map<std::string, std::string> engineNameMap_;

      // Keep the name of the file where we want to save the state
      // of all declared engines at the end of each event. A blank
      // name means don't bother.  Also, keep a record of whether
      // the save file name has been recorded in the job report.
      std::string saveFileName_;
      bool saveFileNameRecorded_;
      std::ofstream outFile_;

      // Keep the name of the file from which we restore the state
      // of all declared engines at the beginning of a run. A
      // blank name means there isn't one.
      std::string restoreFileName_;

      // This turns on or off the checks that ensure no random
      // numbers are generated in a module during construction,
      // beginJob, beginRun, endLuminosityBlock, endRun or endJob.
      bool enableChecking_;

      // True before the first beginLumi call
      bool firstLumi_;

      // In a multiprocess job this will have the index of the child process
      // incremented by one as each child is forked
      unsigned childIndex_;

      uint32_t eventSeedOffset_;

      bool failedToFindStatesInLumi_;

      static uint32_t maxSeedRanecu;
      static uint32_t maxSeedHepJames;
      static uint32_t maxSeedTRandom3;
    };
  }
}

#endif
