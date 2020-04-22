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
#include "FWCore/Utilities/interface/get_underlying_safe.h"

#include <atomic>
#include <cstdint>
#include <fstream>
#include <iosfwd>
#include <istream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

class RandomEngineState;

namespace CLHEP {
  class HepRandomEngine;
}

namespace edm {
  class ActivityRegistry;
  class ConfigurationDescriptions;
  class ConsumesCollector;
  class Event;
  class LuminosityBlock;
  class LuminosityBlockIndex;
  class ModuleCallingContext;
  class ModuleDescription;
  class ParameterSet;
  class StreamContext;
  class StreamID;

  namespace service {

    class SystemBounds;

    class RandomNumberGeneratorService : public RandomNumberGenerator {
    public:
      RandomNumberGeneratorService(ParameterSet const& pset, ActivityRegistry& activityRegistry);
      ~RandomNumberGeneratorService() override;

      /// Use the next 2 functions to get the random number engine.
      /// These are the only functions most modules should call.

      /// Use this engine in event methods
      CLHEP::HepRandomEngine& getEngine(StreamID const& streamID) override;

      /// Use this engine in the global begin luminosity block method
      CLHEP::HepRandomEngine& getEngine(LuminosityBlockIndex const& luminosityBlockIndex) override;

      std::unique_ptr<CLHEP::HepRandomEngine> cloneEngine(LuminosityBlockIndex const&) override;

      // This returns the seed from the configuration. In the unusual case where an
      // an engine type takes multiple seeds to initialize a sequence, this function
      // only returns the first. As a general rule, this function should not be used,
      // but is available for backward compatibility and debugging. It might be useful
      // for some types of tests. Using this to seed engines constructed in modules is
      // not recommended because (unless done very carefully) it will create duplicate
      // sequences in different threads and/or data races. Also, if engines are created
      // by modules the replay mechanism will be broken.
      // Because it is dangerous and could be misused, this function might be deleted
      // someday if we ever find time to delete all uses of it in CMSSW. There are of
      // order 10 last time I checked ...
      std::uint32_t mySeed() const override;

      static void fillDescriptions(ConfigurationDescriptions& descriptions);

      void preModuleConstruction(ModuleDescription const& description);
      void preModuleDestruction(ModuleDescription const& description);
      void preallocate(SystemBounds const&);

      void preBeginLumi(LuminosityBlock const& lumi) override;
      void postEventRead(Event const& event) override;
      void setLumiCache(LuminosityBlockIndex, std::vector<RandomEngineState> const& iStates) override;
      void setEventCache(StreamID, std::vector<RandomEngineState> const& iStates) override;

      /// These next 12 functions are only used to check that random numbers are not
      /// being generated in these methods when enable checking is configured on.
      void preModuleBeginStream(StreamContext const& sc, ModuleCallingContext const& mcc);
      void postModuleBeginStream(StreamContext const& sc, ModuleCallingContext const& mcc);

      void preModuleEndStream(StreamContext const& sc, ModuleCallingContext const& mcc);
      void postModuleEndStream(StreamContext const& sc, ModuleCallingContext const& mcc);

      void preModuleStreamBeginRun(StreamContext const& sc, ModuleCallingContext const& mcc);
      void postModuleStreamBeginRun(StreamContext const& sc, ModuleCallingContext const& mcc);

      void preModuleStreamEndRun(StreamContext const& sc, ModuleCallingContext const& mcc);
      void postModuleStreamEndRun(StreamContext const& sc, ModuleCallingContext const& mcc);

      void preModuleStreamBeginLumi(StreamContext const& sc, ModuleCallingContext const& mcc);
      void postModuleStreamBeginLumi(StreamContext const& sc, ModuleCallingContext const& mcc);

      void preModuleStreamEndLumi(StreamContext const& sc, ModuleCallingContext const& mcc);
      void postModuleStreamEndLumi(StreamContext const& sc, ModuleCallingContext const& mcc);

      /// These two are used by the RandomEngineStateProducer
      std::vector<RandomEngineState> const& getLumiCache(LuminosityBlockIndex const&) const override;
      std::vector<RandomEngineState> const& getEventCache(StreamID const&) const override;

      void consumes(ConsumesCollector&& iC) const override;

      /// For debugging
      void print(std::ostream& os) const override;

    private:
      typedef std::vector<std::uint32_t> VUint32;

      class LabelAndEngine {
      public:
        LabelAndEngine(std::string const& theLabel,
                       VUint32 const& theSeeds,
                       std::shared_ptr<CLHEP::HepRandomEngine> const& theEngine)
            : label_(theLabel), seeds_(theSeeds), engine_(theEngine) {}
        std::string const& label() const { return label_; }
        VUint32 const& seeds() const { return seeds_; }
        std::shared_ptr<CLHEP::HepRandomEngine const> engine() const { return get_underlying_safe(engine_); }
        std::shared_ptr<CLHEP::HepRandomEngine>& engine() { return get_underlying_safe(engine_); }
        void setSeed(std::uint32_t v, unsigned int index) { seeds_.at(index) = v; }

      private:
        std::string label_;
        VUint32 seeds_;
        edm::propagate_const<std::shared_ptr<CLHEP::HepRandomEngine>> engine_;
      };

      // This class exists because it is faster to lookup a module using
      // the moduleID (an integer) than the label (a string). There is a
      // one to one association between LabelAndEngine objects and ModuleIDToEngine objects.
      class ModuleIDToEngine {
      public:
        ModuleIDToEngine(LabelAndEngine* theLabelAndEngine, unsigned int theModuleID)
            : engineState_(), labelAndEngine_(theLabelAndEngine), moduleID_(theModuleID) {}

        std::vector<unsigned long> const& engineState() const { return engineState_; }
        LabelAndEngine const* labelAndEngine() const { return get_underlying_safe(labelAndEngine_); }
        LabelAndEngine*& labelAndEngine() { return get_underlying_safe(labelAndEngine_); }
        unsigned int moduleID() const { return moduleID_; }
        void setEngineState(std::vector<unsigned long> const& v) { engineState_ = v; }
        // Used to sort so binary lookup can be used on a container of these.
        bool operator<(ModuleIDToEngine const& r) const { return moduleID() < r.moduleID(); }

      private:
        std::vector<unsigned long> engineState_;  // Used only for check in stream transitions
        edm::propagate_const<LabelAndEngine*> labelAndEngine_;
        unsigned int moduleID_;
      };

      RandomNumberGeneratorService(RandomNumberGeneratorService const&) = delete;
      RandomNumberGeneratorService const& operator=(RandomNumberGeneratorService const&) = delete;

      void preModuleStreamCheck(StreamContext const& sc, ModuleCallingContext const& mcc);
      void postModuleStreamCheck(StreamContext const& sc, ModuleCallingContext const& mcc);

      void readFromLuminosityBlock(LuminosityBlock const& lumi);
      void readFromEvent(Event const& event);

      void snapShot(std::vector<LabelAndEngine> const& engines, std::vector<RandomEngineState>& cache);
      void restoreFromCache(std::vector<RandomEngineState> const& cache, std::vector<LabelAndEngine>& engines);

      void checkEngineType(std::string const& typeFromConfig,
                           std::string const& typeFromEvent,
                           std::string const& engineLabel) const;

      void saveStatesToFile(std::string const& fileName,
                            StreamID const& streamID,
                            LuminosityBlockIndex const& lumiIndex);
      void writeStates(std::vector<RandomEngineState> const& v, std::ofstream& outFile);
      void writeVector(VUint32 const& v, std::ofstream& outFile);
      std::string constructSaveFileName() const;

      void readEventStatesFromTextFile(std::string const& fileName, std::vector<RandomEngineState>& cache);
      void readLumiStatesFromTextFile(std::string const& fileName, std::vector<RandomEngineState>& cache);
      void readStatesFromFile(std::string const& fileName,
                              std::vector<RandomEngineState>& cache,
                              std::string const& whichStates);
      bool readEngineState(std::istream& is,
                           std::vector<RandomEngineState>& cache,
                           std::string const& whichStates,
                           bool& saveToCache);
      void readVector(std::istream& is, unsigned numItems, std::vector<std::uint32_t>& v);

      void createEnginesInVector(std::vector<LabelAndEngine>& engines,
                                 unsigned int seedOffset,
                                 unsigned int eventSeedOffset,
                                 std::vector<ModuleIDToEngine>& moduleIDVector);

      void resetEngineSeeds(LabelAndEngine& labelAndEngine,
                            std::string const& engineName,
                            VUint32 const& seeds,
                            std::uint32_t offset1,
                            std::uint32_t offset2);

      // ---------- member data --------------------------------

      unsigned int nStreams_;

      // This exists because we can look things up faster using the moduleID
      // than using string comparisons with the moduleLabel
      std::vector<std::vector<ModuleIDToEngine>> streamModuleIDToEngine_;  // streamID, sorted by moduleID
      std::vector<std::vector<ModuleIDToEngine>> lumiModuleIDToEngine_;    // luminosityBlockIndex, sortedByModuleID

      // Holds the engines, plus the seeds and module label also
      std::vector<std::vector<LabelAndEngine>> streamEngines_;  // streamID, sorted by label
      std::vector<std::vector<LabelAndEngine>> lumiEngines_;    // luminosityBlockIndex, sorted by label

      // These hold the input tags needed to retrieve the states
      // of the random number engines stored in a previous process.
      // If the label in the tag is the empty string (the default),
      // then the service does not try to restore the random numbers.
      edm::InputTag restoreStateTag_;
      edm::InputTag restoreStateBeginLumiTag_;

      std::vector<std::vector<RandomEngineState>> eventCache_;  // streamID, sorted by module label
      std::vector<std::vector<RandomEngineState>> lumiCache_;   // luminosityBlockIndex, sorted by module label

      // This is used to keep track of the seeds and engine name from
      // the configuration. The map key is the module label.
      // The module ID is filled in as modules are constructed.
      // It is left as max unsigned if the module is never constructed and not in the process
      class SeedsAndName {
      public:
        static constexpr unsigned int kInvalid = std::numeric_limits<unsigned int>::max();

        SeedsAndName(VUint32 const& theSeeds, std::string const& theEngineName)
            : seeds_(theSeeds), engineName_(theEngineName), moduleID_(kInvalid) {}
        VUint32 const& seeds() const { return seeds_; }
        std::string const& engineName() const { return engineName_; }
        unsigned int moduleID() const { return moduleID_; }
        void setModuleID(unsigned int v) { moduleID_ = v; }

      private:
        VUint32 seeds_;
        std::string engineName_;
        unsigned int moduleID_;
      };
      std::map<std::string, SeedsAndName> seedsAndNameMap_;

      // Keep the name of the file where we want to save the state
      // of all declared engines at the end of each event. A blank
      // name means don't bother.  Also, keep a record of whether
      // the save file name has been recorded in the job report.
      std::string saveFileName_;
      std::atomic<bool> saveFileNameRecorded_;
      std::vector<edm::propagate_const<std::shared_ptr<std::ofstream>>> outFiles_;  // streamID

      // Keep the name of the file from which we restore the state
      // of all declared engines at the beginning of a run. A
      // blank name means there isn't one.
      std::string restoreFileName_;

      // This turns on or off the checks that ensure no random
      // numbers are generated in a module during stream
      // beginStream, beginRun, endLuminosityBlock, or endRun.
      bool enableChecking_;

      std::uint32_t eventSeedOffset_;

      bool verbose_;

      static const std::vector<std::uint32_t>::size_type maxSeeds;
      static const std::vector<std::uint32_t>::size_type maxStates;
      static const std::uint32_t maxSeedRanecu;
      static const std::uint32_t maxSeedHepJames;
      static const std::uint32_t maxSeedTRandom3;
    };
  }  // namespace service
}  // namespace edm
#endif
