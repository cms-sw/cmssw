// -*- C++ -*-
//
// Package:     RandomEngine
// Class  :     RandomNumberGeneratorService
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones, W. David Dagenhart
//   Created:  Tue Mar  7 09:43:46 EST 2006 (originally in FWCore/Services)
//

#include "IOMC/RandomEngine/src/RandomNumberGeneratorService.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterWildcard.h"
#include "FWCore/ServiceRegistry/interface/CurrentModuleOnThread.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/LuminosityBlockIndex.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "IOMC/RandomEngine/src/TRandomAdaptor.h"
#include "SimDataFormats/RandomEngine/interface/RandomEngineState.h"
#include "SimDataFormats/RandomEngine/interface/RandomEngineStates.h"

#include "CLHEP/Random/engineIDulong.h"
#include "CLHEP/Random/JamesRandom.h"
#include "CLHEP/Random/RanecuEngine.h"
#include "CLHEP/Random/MixMaxRng.h"

#include <algorithm>
#include <cassert>
#include <ostream>
#include <sstream>
#include <unistd.h>

namespace edm {
  namespace service {

    const std::vector<std::uint32_t>::size_type RandomNumberGeneratorService::maxSeeds = 65536U;
    const std::vector<std::uint32_t>::size_type RandomNumberGeneratorService::maxStates = 65536U;
    const std::uint32_t RandomNumberGeneratorService::maxSeedRanecu = 2147483647U;
    const std::uint32_t RandomNumberGeneratorService::maxSeedHepJames = 900000000U;
    const std::uint32_t RandomNumberGeneratorService::maxSeedTRandom3 = 4294967295U;

    RandomNumberGeneratorService::RandomNumberGeneratorService(ParameterSet const& pset,
                                                               ActivityRegistry& activityRegistry)
        : nStreams_(0),
          saveFileName_(pset.getUntrackedParameter<std::string>("saveFileName")),
          saveFileNameRecorded_(false),
          restoreFileName_(pset.getUntrackedParameter<std::string>("restoreFileName")),
          enableChecking_(pset.getUntrackedParameter<bool>("enableChecking")),
          eventSeedOffset_(pset.getUntrackedParameter<unsigned>("eventSeedOffset")),
          verbose_(pset.getUntrackedParameter<bool>("verbose")) {
      if (pset.exists("restoreStateTag")) {
        restoreStateTag_ = pset.getUntrackedParameter<edm::InputTag>("restoreStateTag");
        if (restoreStateTag_.process().empty()) {
          restoreStateTag_ = edm::InputTag(restoreStateTag_.label(), "", edm::InputTag::kSkipCurrentProcess);
        }
      } else {
        restoreStateTag_ = edm::InputTag(
            pset.getUntrackedParameter<std::string>("restoreStateLabel"), "", edm::InputTag::kSkipCurrentProcess);
      }
      restoreStateBeginLumiTag_ = edm::InputTag(restoreStateTag_.label(), "beginLumi", restoreStateTag_.process());

      if (!restoreFileName_.empty() && !restoreStateTag_.label().empty()) {
        throw Exception(errors::Configuration) << "In the configuration for the RandomNumberGeneratorService both\n"
                                               << "restoreFileName and restoreStateLabel were set to nonempty values\n"
                                               << "which is illegal.  It is impossible to restore the random engine\n"
                                               << "states two different ways in the same process.\n";
      }

      // The saveFileName must correspond to a file name without any path specification.
      // Throw if that is not true.
      if (!saveFileName_.empty() && (saveFileName_.find('/') != std::string::npos)) {
        throw Exception(errors::Configuration)
            << "The saveFileName parameter must be a simple file name with no path\n"
            << "specification. In the configuration, it was given the value \"" << saveFileName_ << "\"\n";
      }

      std::uint32_t initialSeed;
      VUint32 initialSeedSet;
      std::string engineName;

      std::vector<std::string> pSets = pset.getParameterNamesForType<ParameterSet>();
      for (auto const& label : pSets) {
        ParameterSet const& modulePSet = pset.getParameterSet(label);
        engineName = modulePSet.getUntrackedParameter<std::string>("engineName", std::string("HepJamesRandom"));

        bool initialSeedExists = modulePSet.exists("initialSeed");
        bool initialSeedSetExists = modulePSet.exists("initialSeedSet");

        if (initialSeedExists && initialSeedSetExists) {
          throw Exception(errors::Configuration) << "For the module with the label \"" << label << "\",\n"
                                                 << "both the parameters \"initialSeed\" and \"initialSeedSet\"\n"
                                                 << "have been set in the configuration. You must set one or\n"
                                                 << "the other.  It is illegal to set both.\n";
        } else if (!initialSeedExists && !initialSeedSetExists) {
          throw Exception(errors::Configuration) << "For the module with the label \"" << label << "\",\n"
                                                 << "neither the parameter \"initialSeed\" nor \"initialSeedSet\"\n"
                                                 << "has been set in the configuration. You must set one or\n"
                                                 << "the other.\n";
        } else if (initialSeedExists) {
          initialSeed = modulePSet.getUntrackedParameter<std::uint32_t>("initialSeed");
          initialSeedSet.clear();
          initialSeedSet.push_back(initialSeed);
        } else if (initialSeedSetExists) {
          initialSeedSet = modulePSet.getUntrackedParameter<VUint32>("initialSeedSet");
        }
        seedsAndNameMap_.insert(std::pair<std::string, SeedsAndName>(label, SeedsAndName(initialSeedSet, engineName)));

        // For the CLHEP::RanecuEngine case, require a seed set containing exactly two seeds.
        if (engineName == std::string("RanecuEngine")) {
          if (initialSeedSet.size() != 2U) {
            throw Exception(errors::Configuration)
                << "Random engines of type \"RanecuEngine\" require 2 seeds\n"
                << "be specified with the parameter named \"initialSeedSet\".\n"
                << "Either \"initialSeedSet\" was not in the configuration\n"
                << "or its size was not 2 for the module with label \"" << label << "\".\n";
          }
          if (initialSeedSet[0] > maxSeedRanecu ||
              initialSeedSet[1] > maxSeedRanecu) {  // They need to fit in a 31 bit integer
            throw Exception(errors::Configuration)
                << "The RanecuEngine seeds should be in the range 0 to " << maxSeedRanecu << ".\n"
                << "The seeds passed to the RandomNumberGenerationService from the\n"
                   "configuration file were "
                << initialSeedSet[0] << " and " << initialSeedSet[1] << "\nThis was for the module with label \""
                << label << "\".\n";
          }
        }
        // For the other engines, one seed is required
        else {
          if (initialSeedSet.size() != 1U) {
            throw Exception(errors::Configuration)
                << "Random engines of type \"HepJamesRandom\", \"TRandom3\" and \"MixMaxRng\" \n"
                << "require exactly 1 seed be specified in the configuration.\n"
                << "There were " << initialSeedSet.size() << " seeds set for the\n"
                << "module with label \"" << label << "\".\n";
          }
          if (engineName == "HepJamesRandom") {
            if (initialSeedSet[0] > maxSeedHepJames) {
              throw Exception(errors::Configuration)
                  << "The CLHEP::HepJamesRandom engine seed should be in the range 0 to " << maxSeedHepJames << ".\n"
                  << "The seed passed to the RandomNumberGenerationService from the\n"
                     "configuration file was "
                  << initialSeedSet[0] << ".  This was for \n"
                  << "the module with label " << label << ".\n";
            }
          } else if (engineName == "MixMaxRng") {
            if (initialSeedSet[0] > maxSeedTRandom3) {
              throw Exception(errors::Configuration)
                  << "The CLHEP::MixMaxRng engine seed should be in the range 0 to " << maxSeedTRandom3 << ".\n"
                  << "The seed passed to the RandomNumberGenerationService from the\n"
                     "configuration file was "
                  << initialSeedSet[0] << ".  This was for \n"
                  << "the module with label " << label << ".\n";
            }
          } else if (engineName == "TRandom3") {
            if (initialSeedSet[0] > maxSeedTRandom3) {
              throw Exception(errors::Configuration)
                  << "The CLHEP::MixMaxRng engine seed should be in the range 0 to " << maxSeedTRandom3 << ".\n"
                  << "The seed passed to the RandomNumberGenerationService from the\n"
                     "configuration file was "
                  << initialSeedSet[0] << ".  This was for \n"
                  << "the module with label " << label << ".\n";
            }
          } else {
            throw Exception(errors::Configuration)
                << "The random engine name, \"" << engineName << "\", does not correspond to a supported engine.\n"
                << "This engine was configured for the module with label \"" << label << "\"";
          }
        }
      }
      activityRegistry.watchPreModuleConstruction(this, &RandomNumberGeneratorService::preModuleConstruction);
      activityRegistry.watchPreModuleDestruction(this, &RandomNumberGeneratorService::preModuleDestruction);

      activityRegistry.watchPreallocate(this, &RandomNumberGeneratorService::preallocate);

      if (enableChecking_) {
        activityRegistry.watchPreModuleBeginStream(this, &RandomNumberGeneratorService::preModuleBeginStream);
        activityRegistry.watchPostModuleBeginStream(this, &RandomNumberGeneratorService::postModuleBeginStream);

        activityRegistry.watchPreModuleEndStream(this, &RandomNumberGeneratorService::preModuleEndStream);
        activityRegistry.watchPostModuleEndStream(this, &RandomNumberGeneratorService::postModuleEndStream);

        activityRegistry.watchPreModuleStreamBeginRun(this, &RandomNumberGeneratorService::preModuleStreamBeginRun);
        activityRegistry.watchPostModuleStreamBeginRun(this, &RandomNumberGeneratorService::postModuleStreamBeginRun);

        activityRegistry.watchPreModuleStreamEndRun(this, &RandomNumberGeneratorService::preModuleStreamEndRun);
        activityRegistry.watchPostModuleStreamEndRun(this, &RandomNumberGeneratorService::postModuleStreamEndRun);

        activityRegistry.watchPreModuleStreamBeginLumi(this, &RandomNumberGeneratorService::preModuleStreamBeginLumi);
        activityRegistry.watchPostModuleStreamBeginLumi(this, &RandomNumberGeneratorService::postModuleStreamBeginLumi);

        activityRegistry.watchPreModuleStreamEndLumi(this, &RandomNumberGeneratorService::preModuleStreamEndLumi);
        activityRegistry.watchPostModuleStreamEndLumi(this, &RandomNumberGeneratorService::postModuleStreamEndLumi);
      }
    }

    RandomNumberGeneratorService::~RandomNumberGeneratorService() {}

    void RandomNumberGeneratorService::consumes(ConsumesCollector&& iC) const {
      iC.consumes<RandomEngineStates, InLumi>(restoreStateBeginLumiTag_);
      iC.consumes<RandomEngineStates>(restoreStateTag_);
    }

    CLHEP::HepRandomEngine& RandomNumberGeneratorService::getEngine(StreamID const& streamID) {
      ModuleCallingContext const* mcc = CurrentModuleOnThread::getCurrentModuleOnThread();
      if (mcc == nullptr) {
        throw Exception(errors::LogicError)
            << "RandomNumberGeneratorService::getEngine\n"
               "Requested a random number engine from the RandomNumberGeneratorService\n"
               "when no module was active. ModuleCallingContext is null\n";
      }
      unsigned int moduleID = mcc->moduleDescription()->id();

      std::vector<ModuleIDToEngine>& moduleIDVector = streamModuleIDToEngine_.at(streamID.value());
      ModuleIDToEngine target(nullptr, moduleID);
      std::vector<ModuleIDToEngine>::iterator iter =
          std::lower_bound(moduleIDVector.begin(), moduleIDVector.end(), target);
      if (iter == moduleIDVector.end() || iter->moduleID() != moduleID) {
        throw Exception(errors::Configuration)
            << "The module with label \"" << mcc->moduleDescription()->moduleLabel()
            << "\" requested a random number engine from the \n"
               "RandomNumberGeneratorService, but that module was not configured\n"
               "for random numbers.  An engine is created only if a seed(s) is provided\n"
               "in the configuration file.  Please add the following PSet to the\n"
               "configuration file for the RandomNumberGeneratorService:\n\n"
               "  "
            << mcc->moduleDescription()->moduleLabel()
            << " = cms.PSet(\n"
               "    initialSeed = cms.untracked.uint32(your_seed),\n"
               "    engineName = cms.untracked.string('TRandom3')\n"
               "  )\n"
               "where you replace \"your_seed\" with a number and add a comma if necessary\n"
               "The \"engineName\" parameter is optional. If absent the default is \"HepJamesRandom\".\n";
      }
      return *iter->labelAndEngine()->engine();
    }

    CLHEP::HepRandomEngine& RandomNumberGeneratorService::getEngine(LuminosityBlockIndex const& lumiIndex) {
      ModuleCallingContext const* mcc = CurrentModuleOnThread::getCurrentModuleOnThread();
      if (mcc == nullptr) {
        throw Exception(errors::LogicError)
            << "RandomNumberGeneratorService::getEngine\n"
               "Requested a random number engine from the RandomNumberGeneratorService\n"
               "when no module was active. ModuleCallingContext is null\n";
      }
      unsigned int moduleID = mcc->moduleDescription()->id();

      std::vector<ModuleIDToEngine>& moduleIDVector = lumiModuleIDToEngine_.at(lumiIndex.value());
      ModuleIDToEngine target(nullptr, moduleID);
      std::vector<ModuleIDToEngine>::iterator iter =
          std::lower_bound(moduleIDVector.begin(), moduleIDVector.end(), target);
      if (iter == moduleIDVector.end() || iter->moduleID() != moduleID) {
        throw Exception(errors::Configuration)
            << "The module with label \"" << mcc->moduleDescription()->moduleLabel()
            << "\" requested a random number engine from the \n"
               "RandomNumberGeneratorService, but that module was not configured\n"
               "for random numbers.  An engine is created only if a seed(s) is provided\n"
               "in the configuration file.  Please add the following PSet to the\n"
               "configuration file for the RandomNumberGeneratorService:\n\n"
               "  "
            << mcc->moduleDescription()->moduleLabel()
            << " = cms.PSet(\n"
               "    initialSeed = cms.untracked.uint32(your_seed),\n"
               "    engineName = cms.untracked.string('TRandom3')\n"
               "  )\n"
               "where you replace \"your_seed\" with a number and add a comma if necessary\n"
               "The \"engineName\" parameter is optional. If absent the default is \"HepJamesRandom\".\n";
      }
      return *iter->labelAndEngine()->engine();
    }

    std::unique_ptr<CLHEP::HepRandomEngine> RandomNumberGeneratorService::cloneEngine(
        LuminosityBlockIndex const& lumiIndex) {
      CLHEP::HepRandomEngine& existingEngine = getEngine(lumiIndex);

      std::vector<unsigned long> stateL = existingEngine.put();
      long seedL = existingEngine.getSeed();
      std::unique_ptr<CLHEP::HepRandomEngine> newEngine;
      if (stateL[0] == CLHEP::engineIDulong<CLHEP::HepJamesRandom>()) {
        newEngine = std::make_unique<CLHEP::HepJamesRandom>(seedL);
      } else if (stateL[0] == CLHEP::engineIDulong<CLHEP::RanecuEngine>()) {
        newEngine = std::make_unique<CLHEP::RanecuEngine>();
      } else if (stateL[0] == CLHEP::engineIDulong<CLHEP::MixMaxRng>()) {
        newEngine = std::make_unique<CLHEP::MixMaxRng>(seedL);
      } else if (stateL[0] == CLHEP::engineIDulong<TRandomAdaptor>()) {
        newEngine = std::make_unique<TRandomAdaptor>(seedL);
      } else {
        // Sanity check, it should not be possible for this to happen.
        throw Exception(errors::Unknown) << "The RandomNumberGeneratorService is trying to clone unknown engine type\n";
      }
      newEngine->get(stateL);
      return newEngine;
    }

    // PROBABLY TO BE DELETED, This returns the configured seed without
    // any of the modifications for streams or the offset configuration
    // parameter. Maybe useful to use for debugging/checks, but dangerous if one tries
    // to create your own engines using it. It is difficult to get the offsets
    // for streams/forking/offset parameters correct and almost certainly would break
    // replay.
    std::uint32_t RandomNumberGeneratorService::mySeed() const {
      std::string label;
      ModuleCallingContext const* mcc = CurrentModuleOnThread::getCurrentModuleOnThread();
      if (mcc == nullptr) {
        throw Exception(errors::LogicError)
            << "RandomNumberGeneratorService::getEngine()\n"
               "Requested a random number engine from the RandomNumberGeneratorService\n"
               "from an unallowed transition. ModuleCallingContext is null\n";
      } else {
        label = mcc->moduleDescription()->moduleLabel();
      }

      std::map<std::string, SeedsAndName>::const_iterator iter = seedsAndNameMap_.find(label);
      if (iter == seedsAndNameMap_.end()) {
        throw Exception(errors::Configuration)
            << "The module with label \"" << label
            << "\" requested a random number seed from the \n"
               "RandomNumberGeneratorService, but that module was not configured\n"
               "for random numbers.  An engine is created only if a seed(s) is provided\n"
               "in the configuration file.  Please add the following PSet to the\n"
               "configuration file for the RandomNumberGeneratorService:\n\n"
               "  "
            << label
            << " = cms.PSet(\n"
               "    initialSeed = cms.untracked.uint32(your_seed),\n"
               "    engineName = cms.untracked.string('TRandom3')\n"
               "  )\n"
               "where you replace \"your_seed\" with a number and add a comma if necessary\n"
               "The \"engineName\" parameter is optional. If absent the default is \"HepJamesRandom\".\n";
      }
      return iter->second.seeds()[0];
    }

    void RandomNumberGeneratorService::fillDescriptions(ConfigurationDescriptions& descriptions) {
      ParameterSetDescription desc;

      std::string emptyString;
      edm::InputTag emptyInputTag("", "", "");

      desc.addNode(edm::ParameterDescription<edm::InputTag>("restoreStateTag", emptyInputTag, false) xor
                   edm::ParameterDescription<std::string>("restoreStateLabel", emptyString, false));

      desc.addUntracked<std::string>("saveFileName", emptyString);
      desc.addUntracked<std::string>("restoreFileName", emptyString);
      desc.addUntracked<bool>("enableChecking", false);
      desc.addUntracked<unsigned>("eventSeedOffset", 0U);
      desc.addUntracked<bool>("verbose", false);

      ParameterSetDescription val;
      val.addOptionalUntracked<std::uint32_t>("initialSeed");
      val.addOptionalUntracked<std::vector<std::uint32_t> >("initialSeedSet");
      val.addOptionalUntracked<std::string>("engineName");

      ParameterWildcard<ParameterSetDescription> wnode("*", RequireZeroOrMore, true, val);
      wnode.setComment("The name of each ParameterSet will be the associated module label.");
      desc.addNode(wnode);

      descriptions.add("RandomNumberGeneratorService", desc);
    }

    void RandomNumberGeneratorService::preModuleConstruction(ModuleDescription const& description) {
      std::map<std::string, SeedsAndName>::iterator iter = seedsAndNameMap_.find(description.moduleLabel());
      if (iter != seedsAndNameMap_.end()) {
        iter->second.setModuleID(description.id());
      }
    }

    void RandomNumberGeneratorService::preModuleDestruction(ModuleDescription const& description) {
      std::map<std::string, SeedsAndName>::iterator iter = seedsAndNameMap_.find(description.moduleLabel());
      if (iter != seedsAndNameMap_.end()) {
        iter->second.setModuleID(SeedsAndName::kInvalid);
      }
    }

    void RandomNumberGeneratorService::preallocate(SystemBounds const& sb) {
      nStreams_ = sb.maxNumberOfStreams();
      assert(nStreams_ >= 1);
      if (!restoreFileName_.empty() && nStreams_ != 1) {
        throw Exception(errors::Configuration)
            << "Configuration is illegal. The RandomNumberGeneratorService is configured\n"
            << "to run replay using a text file to input the random engine states and\n"
            << "the number of streams is greater than 1. Either set the\n"
            << "parameter named \"restoreFileName\" in the RandomNumberGeneratorService\n"
            << "to the empty string or set the parameter \"numberOfStreams\" in the top\n"
            << "level options parameter set to 1. (Probably these are the default values\n"
            << "and just not setting the parameters will also work)\n";
      }
      unsigned int nConcurrentLumis = sb.maxNumberOfConcurrentLuminosityBlocks();

      streamModuleIDToEngine_.resize(nStreams_);
      lumiModuleIDToEngine_.resize(nConcurrentLumis);
      streamEngines_.resize(nStreams_);
      lumiEngines_.resize(nConcurrentLumis);
      eventCache_.resize(nStreams_);
      lumiCache_.resize(nConcurrentLumis);
      outFiles_.resize(nStreams_);

      for (unsigned int iStream = 0; iStream < nStreams_; ++iStream) {
        unsigned int seedOffset = iStream;
        createEnginesInVector(streamEngines_[iStream], seedOffset, eventSeedOffset_, streamModuleIDToEngine_[iStream]);
        if (!saveFileName_.empty()) {
          outFiles_[iStream] = std::make_shared<std::ofstream>();  // propagate_const<T> has no reset() function
        }
      }
      for (unsigned int iLumi = 0; iLumi < nConcurrentLumis; ++iLumi) {
        unsigned int seedOffset = nStreams_;
        createEnginesInVector(lumiEngines_[iLumi], seedOffset, 0, lumiModuleIDToEngine_[iLumi]);
        snapShot(lumiEngines_[iLumi], lumiCache_[iLumi]);
        if (!restoreFileName_.empty()) {
          readLumiStatesFromTextFile(restoreFileName_, lumiCache_[iLumi]);
        }
      }

      if (!restoreFileName_.empty()) {
        // There is guaranteed to be one stream in this case
        snapShot(streamEngines_[0], eventCache_[0]);
        readEventStatesFromTextFile(restoreFileName_, eventCache_[0]);
        restoreFromCache(eventCache_[0], streamEngines_[0]);
      }
      if (verbose_) {
        print(std::cout);
      }
    }

    void RandomNumberGeneratorService::preBeginLumi(LuminosityBlock const& lumi) {
      if (!restoreStateTag_.label().empty()) {
        // Copy from a product in the LuminosityBlock to cache for a particular luminosityBlockIndex
        readFromLuminosityBlock(lumi);
      }
      // Copy from cache to engine the state for a particular luminosityBlockIndex
      restoreFromCache(lumiCache_[lumi.index()], lumiEngines_[lumi.index()]);
    }

    void RandomNumberGeneratorService::postEventRead(Event const& event) {
      if (!restoreStateTag_.label().empty()) {
        // This initializes the cache before readFromEvent
        snapShot(streamEngines_[event.streamID()], eventCache_[event.streamID()]);

        // copy from Event to event cache
        readFromEvent(event);

        // copy from event cache to engines
        restoreFromCache(eventCache_[event.streamID()], streamEngines_[event.streamID()]);

      } else {
        // copy from engines to event cache
        snapShot(streamEngines_[event.streamID()], eventCache_[event.streamID()]);
      }

      // if requested write text file from both caches
      if (!saveFileName_.empty()) {
        saveStatesToFile(saveFileName_, event.streamID(), event.getLuminosityBlock().index());
        bool expected = false;
        if (saveFileNameRecorded_.compare_exchange_strong(expected, true)) {
          std::string fullName = constructSaveFileName();
          Service<JobReport> reportSvc;
          reportSvc->reportRandomStateFile(fullName);
        }
      }
    }

    void RandomNumberGeneratorService::setLumiCache(LuminosityBlockIndex iLumi,
                                                    std::vector<RandomEngineState> const& iStates) {
      lumiCache_[iLumi] = iStates;
      // Copy from cache to engine the state for a particular luminosityBlockIndex
      restoreFromCache(lumiCache_[iLumi], lumiEngines_[iLumi]);
    }
    void RandomNumberGeneratorService::setEventCache(StreamID iStream, std::vector<RandomEngineState> const& iStates) {
      eventCache_[iStream] = iStates;
      // copy from event cache to engines
      restoreFromCache(eventCache_[iStream], streamEngines_[iStream]);
    }

    void RandomNumberGeneratorService::preModuleBeginStream(StreamContext const& sc, ModuleCallingContext const& mcc) {
      preModuleStreamCheck(sc, mcc);
    }

    void RandomNumberGeneratorService::postModuleBeginStream(StreamContext const& sc, ModuleCallingContext const& mcc) {
      postModuleStreamCheck(sc, mcc);
    }

    void RandomNumberGeneratorService::preModuleEndStream(StreamContext const& sc, ModuleCallingContext const& mcc) {
      preModuleStreamCheck(sc, mcc);
    }

    void RandomNumberGeneratorService::postModuleEndStream(StreamContext const& sc, ModuleCallingContext const& mcc) {
      postModuleStreamCheck(sc, mcc);
    }

    void RandomNumberGeneratorService::preModuleStreamBeginRun(StreamContext const& sc,
                                                               ModuleCallingContext const& mcc) {
      preModuleStreamCheck(sc, mcc);
    }

    void RandomNumberGeneratorService::postModuleStreamBeginRun(StreamContext const& sc,
                                                                ModuleCallingContext const& mcc) {
      postModuleStreamCheck(sc, mcc);
    }

    void RandomNumberGeneratorService::preModuleStreamEndRun(StreamContext const& sc, ModuleCallingContext const& mcc) {
      preModuleStreamCheck(sc, mcc);
    }

    void RandomNumberGeneratorService::postModuleStreamEndRun(StreamContext const& sc,
                                                              ModuleCallingContext const& mcc) {
      postModuleStreamCheck(sc, mcc);
    }

    void RandomNumberGeneratorService::preModuleStreamBeginLumi(StreamContext const& sc,
                                                                ModuleCallingContext const& mcc) {
      preModuleStreamCheck(sc, mcc);
    }

    void RandomNumberGeneratorService::postModuleStreamBeginLumi(StreamContext const& sc,
                                                                 ModuleCallingContext const& mcc) {
      postModuleStreamCheck(sc, mcc);
    }

    void RandomNumberGeneratorService::preModuleStreamEndLumi(StreamContext const& sc,
                                                              ModuleCallingContext const& mcc) {
      preModuleStreamCheck(sc, mcc);
    }

    void RandomNumberGeneratorService::postModuleStreamEndLumi(StreamContext const& sc,
                                                               ModuleCallingContext const& mcc) {
      postModuleStreamCheck(sc, mcc);
    }

    std::vector<RandomEngineState> const& RandomNumberGeneratorService::getLumiCache(
        LuminosityBlockIndex const& lumiIndex) const {
      return lumiCache_.at(lumiIndex.value());
    }

    std::vector<RandomEngineState> const& RandomNumberGeneratorService::getEventCache(StreamID const& streamID) const {
      return eventCache_.at(streamID.value());
    }

    void RandomNumberGeneratorService::print(std::ostream& os) const {
      os << "\n\nRandomNumberGeneratorService dump\n\n";

      os << "    Contents of seedsAndNameMap (label moduleID engineType seeds)\n";
      for (auto const& entry : seedsAndNameMap_) {
        os << "        " << entry.first << "  " << entry.second.moduleID() << "  " << entry.second.engineName();
        for (auto val : entry.second.seeds()) {
          os << "  " << val;
        }
        os << "\n";
      }
      os << "    nStreams_ = " << nStreams_ << "\n";
      os << "    saveFileName_ = " << saveFileName_ << "\n";
      os << "    saveFileNameRecorded_ = " << saveFileNameRecorded_ << "\n";
      os << "    restoreFileName_ = " << restoreFileName_ << "\n";
      os << "    enableChecking_ = " << enableChecking_ << "\n";
      os << "    eventSeedOffset_ = " << eventSeedOffset_ << "\n";
      os << "    verbose_ = " << verbose_ << "\n";
      os << "    restoreStateTag_ = " << restoreStateTag_ << "\n";
      os << "    restoreStateBeginLumiTag_ = " << restoreStateBeginLumiTag_ << "\n";

      os << "\n    streamEngines_\n";
      unsigned int iStream = 0;
      for (auto const& k : streamEngines_) {
        os << "        Stream " << iStream << "\n";
        for (auto const& i : k) {
          os << "        " << i.label();
          for (auto const& j : i.seeds()) {
            os << " " << j;
          }
          os << " " << i.engine()->name();
          if (i.engine()->name() == std::string("HepJamesRandom")) {
            os << "  " << i.engine()->getSeed();
          } else if (i.engine()->name() == std::string("MixMaxRng")) {
            os << "  " << i.engine()->getSeed();
          } else {
            os << "  engine does not know seeds";
          }
          os << "\n";
        }
        ++iStream;
      }
      os << "\n    lumiEngines_\n";
      unsigned int iLumi = 0;
      for (auto const& k : lumiEngines_) {
        os << "        lumiIndex " << iLumi << "\n";
        for (auto const& i : k) {
          os << "        " << i.label();
          for (auto const& j : i.seeds()) {
            os << " " << j;
          }
          os << " " << i.engine()->name();
          if (i.engine()->name() == std::string("HepJamesRandom")) {
            os << "  " << i.engine()->getSeed();
          } else if (i.engine()->name() == std::string("MixMaxRng")) {
            os << "  " << i.engine()->getSeed();
          } else {
            os << "  engine does not know seeds";
          }
          os << "\n";
        }
        ++iLumi;
      }
    }

    void RandomNumberGeneratorService::preModuleStreamCheck(StreamContext const& sc, ModuleCallingContext const& mcc) {
      if (enableChecking_) {
        unsigned int moduleID = mcc.moduleDescription()->id();
        std::vector<ModuleIDToEngine>& moduleIDVector = streamModuleIDToEngine_.at(sc.streamID().value());
        ModuleIDToEngine target(nullptr, moduleID);
        std::vector<ModuleIDToEngine>::iterator iter =
            std::lower_bound(moduleIDVector.begin(), moduleIDVector.end(), target);
        if (iter != moduleIDVector.end() && iter->moduleID() == moduleID) {
          LabelAndEngine* labelAndEngine = iter->labelAndEngine();
          iter->setEngineState(labelAndEngine->engine()->put());
        }
      }
    }

    void RandomNumberGeneratorService::postModuleStreamCheck(StreamContext const& sc, ModuleCallingContext const& mcc) {
      if (enableChecking_) {
        unsigned int moduleID = mcc.moduleDescription()->id();
        std::vector<ModuleIDToEngine>& moduleIDVector = streamModuleIDToEngine_.at(sc.streamID().value());
        ModuleIDToEngine target(nullptr, moduleID);
        std::vector<ModuleIDToEngine>::iterator iter =
            std::lower_bound(moduleIDVector.begin(), moduleIDVector.end(), target);
        if (iter != moduleIDVector.end() && iter->moduleID() == moduleID) {
          LabelAndEngine* labelAndEngine = iter->labelAndEngine();
          if (iter->engineState() != labelAndEngine->engine()->put()) {
            throw Exception(errors::LogicError)
                << "It is illegal to generate random numbers during beginStream, endStream,\n"
                   "beginRun, endRun, beginLumi, endLumi because that makes it very difficult\n"
                   "to replay the processing of individual events.  Random numbers were\n"
                   "generated during one of these methods for the module with class name\n\""
                << mcc.moduleDescription()->moduleName()
                << "\" "
                   "and module label \""
                << mcc.moduleDescription()->moduleLabel() << "\"\n";
          }
        }
      }
    }

    void RandomNumberGeneratorService::readFromLuminosityBlock(LuminosityBlock const& lumi) {
      Service<TriggerNamesService> tns;
      if (tns.isAvailable()) {
        if (tns->getProcessName() == restoreStateTag_.process()) {
          throw Exception(errors::Configuration)
              << "In the configuration for the RandomNumberGeneratorService the\n"
              << "restoreStateTag contains the current process which is illegal.\n"
              << "The process name in the replay process should have been changed\n"
              << "to be different than the original process name and the restoreStateTag\n"
              << "should contain either the original process name or an empty process name.\n";
        }
      }

      Handle<RandomEngineStates> states;
      lumi.getByLabel(restoreStateBeginLumiTag_, states);

      if (!states.isValid()) {
        throw Exception(errors::ProductNotFound)
            << "The RandomNumberGeneratorService is trying to restore\n"
            << "the state of the random engines by reading a product from\n"
            << "the LuminosityBlock with input tag \"" << restoreStateBeginLumiTag_ << "\".\n"
            << "It could not find the product.\n"
            << "Either the product in the LuminosityBlock was dropped or\n"
            << "not produced or the configured input tag is incorrect or there is a bug somewhere\n";
        return;
      }
      states->getRandomEngineStates(lumiCache_.at(lumi.index()));
    }

    void RandomNumberGeneratorService::readFromEvent(Event const& event) {
      Handle<RandomEngineStates> states;

      event.getByLabel(restoreStateTag_, states);

      if (!states.isValid()) {
        throw Exception(errors::ProductNotFound)
            << "The RandomNumberGeneratorService is trying to restore\n"
            << "the state of the random engines by reading a product from\n"
            << "the Event with input tag \"" << restoreStateTag_ << "\".\n"
            << "It could not find the product.\n"
            << "Either the product in the Event was dropped or\n"
            << "not produced or the configured input tag is incorrect or there is a bug somewhere\n";
        return;
      }
      states->getRandomEngineStates(eventCache_.at(event.streamID()));
    }

    void RandomNumberGeneratorService::snapShot(std::vector<LabelAndEngine> const& engines,
                                                std::vector<RandomEngineState>& cache) {
      cache.resize(engines.size());
      std::vector<RandomEngineState>::iterator state = cache.begin();

      for (std::vector<LabelAndEngine>::const_iterator iter = engines.begin(); iter != engines.end(); ++iter, ++state) {
        std::string const& label = iter->label();
        state->setLabel(label);
        state->setSeed(iter->seeds());

        std::vector<unsigned long> stateL = iter->engine()->put();
        state->clearStateVector();
        state->reserveStateVector(stateL.size());
        for (auto element : stateL) {
          state->push_back_stateVector(static_cast<std::uint32_t>(element));
        }
      }
    }

    void RandomNumberGeneratorService::restoreFromCache(std::vector<RandomEngineState> const& cache,
                                                        std::vector<LabelAndEngine>& engines) {
      std::vector<LabelAndEngine>::iterator labelAndEngine = engines.begin();
      for (auto const& cachedState : cache) {
        std::string const& engineLabel = cachedState.getLabel();

        std::vector<std::uint32_t> const& engineState = cachedState.getState();
        std::vector<unsigned long> engineStateL;
        engineStateL.reserve(engineState.size());
        for (auto const& value : engineState) {
          engineStateL.push_back(static_cast<unsigned long>(value));
        }

        std::vector<std::uint32_t> const& engineSeeds = cachedState.getSeed();
        std::vector<long> engineSeedsL;
        engineSeedsL.reserve(engineSeeds.size());
        for (auto const& val : engineSeeds) {
          long seedL = static_cast<long>(val);
          engineSeedsL.push_back(seedL);

          // There is a dangerous conversion from std::uint32_t to long
          // that occurs above. In the next 2 lines we check the
          // behavior is what we need for the service to work
          // properly.  This conversion is forced on us by the
          // CLHEP and ROOT interfaces. If the assert ever starts
          // to fail we will have to come up with a way to deal
          // with this.
          std::uint32_t seedu32 = static_cast<std::uint32_t>(seedL);
          assert(val == seedu32);
        }

        assert(labelAndEngine != engines.end() && engineLabel == labelAndEngine->label());
        std::shared_ptr<CLHEP::HepRandomEngine> const& engine = labelAndEngine->engine();

        // We need to handle each type of engine differently because each
        // has different requirements on the seed or seeds.
        if (engineStateL[0] == CLHEP::engineIDulong<CLHEP::HepJamesRandom>()) {
          checkEngineType(engine->name(), std::string("HepJamesRandom"), engineLabel);

          // These two lines actually restore the seed and engine state.
          engine->setSeed(engineSeedsL[0], 0);
          engine->get(engineStateL);

          labelAndEngine->setSeed(engineSeeds[0], 0);
        } else if (engineStateL[0] == CLHEP::engineIDulong<CLHEP::RanecuEngine>()) {
          checkEngineType(engine->name(), std::string("RanecuEngine"), engineLabel);

          // This line actually restores the engine state.
          engine->get(engineStateL);

          labelAndEngine->setSeed(engineSeeds[0], 0);
          labelAndEngine->setSeed(engineSeeds[1], 1);
        } else if (engineStateL[0] == CLHEP::engineIDulong<CLHEP::MixMaxRng>()) {
          checkEngineType(engine->name(), std::string("MixMaxRng"), engineLabel);

          // This line actually restores the engine state.
          engine->setSeed(engineSeedsL[0], 0);
          engine->get(engineStateL);

          labelAndEngine->setSeed(engineSeeds[0], 0);
        } else if (engineStateL[0] == CLHEP::engineIDulong<TRandomAdaptor>()) {
          checkEngineType(engine->name(), std::string("TRandom3"), engineLabel);

          // This line actually restores the engine state.
          engine->setSeed(engineSeedsL[0], 0);
          engine->get(engineStateL);

          labelAndEngine->setSeed(engineSeeds[0], 0);
        } else {
          // This should not be possible because this code should be able to restore
          // any kind of engine whose state can be saved.
          throw Exception(errors::Unknown)
              << "The RandomNumberGeneratorService is trying to restore the state\n"
                 "of the random engines.  The state in the event indicates an engine\n"
                 "of an unknown type.  This should not be possible unless you are\n"
                 "running with an old code release on a new file that was created\n"
                 "with a newer release which had new engine types added.  In this case\n"
                 "the only solution is to use a newer release.  In any other case, notify\n"
                 "the EDM developers because this should not be possible\n";
        }
        ++labelAndEngine;
      }
    }

    void RandomNumberGeneratorService::checkEngineType(std::string const& typeFromConfig,
                                                       std::string const& typeFromEvent,
                                                       std::string const& engineLabel) const {
      if (typeFromConfig != typeFromEvent) {
        throw Exception(errors::Configuration)
            << "The RandomNumberGeneratorService is trying to restore\n"
            << "the state of the random engine for the module \"" << engineLabel << "\".  An\n"
            << "error was detected because the type of the engine in the\n"
            << "input file and the configuration file do not match.\n"
            << "In the configuration file the type is \"" << typeFromConfig << "\".\nIn the input file the type is \""
            << typeFromEvent << "\".  If\n"
            << "you are not generating any random numbers in this module, then\n"
            << "remove the line in the configuration file that gives it\n"
            << "a seed and the error will go away.  Otherwise, you must give\n"
            << "this module the same engine type in the configuration file or\n"
            << "stop trying to restore the random engine state.\n";
      }
    }

    void RandomNumberGeneratorService::saveStatesToFile(std::string const& fileName,
                                                        StreamID const& streamID,
                                                        LuminosityBlockIndex const& lumiIndex) {
      std::ofstream& outFile = *outFiles_.at(streamID);

      if (!outFile.is_open()) {
        std::stringstream file;
        file << fileName;
        if (nStreams_ > 1) {
          file << "_" << streamID.value();
        }

        outFile.open(file.str().c_str(), std::ofstream::out | std::ofstream::trunc);

        if (!outFile) {
          throw Exception(errors::Configuration)
              << "Unable to open the file \"" << file.str() << "\" to save the state of the random engines.\n";
        }
      }

      outFile.seekp(0, std::ios_base::beg);
      outFile << "<RandomEngineStates>\n";

      outFile << "<Event>\n";
      writeStates(eventCache_.at(streamID), outFile);
      outFile << "</Event>\n";

      outFile << "<Lumi>\n";
      writeStates(lumiCache_.at(lumiIndex), outFile);
      outFile << "</Lumi>\n";

      outFile << "</RandomEngineStates>\n";
      outFile.flush();
    }

    void RandomNumberGeneratorService::writeStates(std::vector<RandomEngineState> const& v, std::ofstream& outFile) {
      for (auto& state : v) {
        std::vector<std::uint32_t> const& seedVector = state.getSeed();
        std::vector<std::uint32_t>::size_type seedVectorLength = seedVector.size();

        std::vector<std::uint32_t> const& stateVector = state.getState();
        std::vector<std::uint32_t>::size_type stateVectorLength = stateVector.size();

        outFile << "<ModuleLabel>\n" << state.getLabel() << "\n</ModuleLabel>\n";

        outFile << "<SeedLength>\n" << seedVectorLength << "\n</SeedLength>\n";
        outFile << "<InitialSeeds>\n";
        writeVector(seedVector, outFile);
        outFile << "</InitialSeeds>\n";
        outFile << "<FullStateLength>\n" << stateVectorLength << "\n</FullStateLength>\n";
        outFile << "<FullState>\n";
        writeVector(stateVector, outFile);
        outFile << "</FullState>\n";
      }
    }

    void RandomNumberGeneratorService::writeVector(VUint32 const& v, std::ofstream& outFile) {
      if (v.empty())
        return;
      size_t numItems = v.size();
      for (size_t i = 0; i < numItems; ++i) {
        if (i != 0 && i % 10 == 0)
          outFile << "\n";
        outFile << std::setw(13) << v[i];
      }
      outFile << "\n";
    }

    std::string RandomNumberGeneratorService::constructSaveFileName() const {
      char directory[1500];
      std::string fullName(getcwd(directory, sizeof(directory)) ? directory : "/PathIsTooBig");
      fullName += "/" + saveFileName_;
      return fullName;
    }

    void RandomNumberGeneratorService::readEventStatesFromTextFile(std::string const& fileName,
                                                                   std::vector<RandomEngineState>& cache) {
      std::string whichStates("<Event>");
      readStatesFromFile(fileName, cache, whichStates);
    }

    void RandomNumberGeneratorService::readLumiStatesFromTextFile(std::string const& fileName,
                                                                  std::vector<RandomEngineState>& cache) {
      std::string whichStates("<Lumi>");
      readStatesFromFile(fileName, cache, whichStates);
    }

    void RandomNumberGeneratorService::readStatesFromFile(std::string const& fileName,
                                                          std::vector<RandomEngineState>& cache,
                                                          std::string const& whichStates) {
      std::ifstream inFile;
      inFile.open(fileName.c_str(), std::ifstream::in);
      if (!inFile) {
        throw Exception(errors::Configuration)
            << "Unable to open the file \"" << fileName << "\" to restore the random engine states.\n";
      }

      std::string text;
      inFile >> text;
      if (!inFile.good() || text != std::string("<RandomEngineStates>")) {
        throw Exception(errors::Configuration)
            << "Attempting to read file with random number engine states.\n"
            << "File \"" << restoreFileName_ << "\" is ill-structured or otherwise corrupted.\n"
            << "Cannot read the file header word.\n";
      }
      bool saveToCache = false;
      while (readEngineState(inFile, cache, whichStates, saveToCache)) {
      }
    }

    bool RandomNumberGeneratorService::readEngineState(std::istream& is,
                                                       std::vector<RandomEngineState>& cache,
                                                       std::string const& whichStates,
                                                       bool& saveToCache) {
      std::string leading;
      std::string trailing;
      std::string moduleLabel;
      std::vector<std::uint32_t>::size_type seedVectorSize;
      std::vector<std::uint32_t> seedVector;
      std::vector<std::uint32_t>::size_type stateVectorSize;
      std::vector<std::uint32_t> stateVector;

      // First we need to look for the special strings
      // that mark the end of the file and beginning and
      // and end of the data for different sections.

      is >> leading;
      if (!is.good()) {
        throw Exception(errors::Configuration)
            << "File \"" << restoreFileName_ << "\" is ill-structured or otherwise corrupted.\n"
            << "Cannot read next field and did not hit the end yet.\n";
      }

      // This marks the end of the file. We are done.
      if (leading == std::string("</RandomEngineStates>"))
        return false;

      // This marks the end of a section of the data
      if (leading == std::string("</Event>") || leading == std::string("</Lumi>")) {
        saveToCache = false;
        return true;
      }

      // This marks the beginning of a section
      if (leading == std::string("<Event>") || leading == std::string("<Lumi>")) {
        saveToCache = (leading == whichStates);
        return true;
      }

      // Process the next engine state

      is >> moduleLabel >> trailing;
      if (!is.good() || leading != std::string("<ModuleLabel>") || trailing != std::string("</ModuleLabel>")) {
        throw Exception(errors::Configuration)
            << "File \"" << restoreFileName_ << "\" is ill-structured or otherwise corrupted.\n"
            << "Cannot read a module label when restoring random engine states.\n";
      }

      is >> leading >> seedVectorSize >> trailing;
      if (!is.good() || leading != std::string("<SeedLength>") || trailing != std::string("</SeedLength>")) {
        throw Exception(errors::Configuration)
            << "File \"" << restoreFileName_ << "\" is ill-structured or otherwise corrupted.\n"
            << "Cannot read seed vector length when restoring random engine states.\n";
      }

      is >> leading;
      if (!is.good() || leading != std::string("<InitialSeeds>")) {
        throw Exception(errors::Configuration)
            << "File \"" << restoreFileName_ << "\" is ill-structured or otherwise corrupted.\n"
            << "Cannot read beginning of InitialSeeds when restoring random engine states.\n";
      }

      if (seedVectorSize > maxSeeds) {
        throw Exception(errors::Configuration)
            << "File \"" << restoreFileName_ << "\" is ill-structured or otherwise corrupted.\n"
            << "The number of seeds exceeds 64K.\n";
      }

      readVector(is, seedVectorSize, seedVector);

      is >> trailing;
      if (!is.good() || trailing != std::string("</InitialSeeds>")) {
        throw Exception(errors::Configuration)
            << "File \"" << restoreFileName_ << "\" is ill-structured or otherwise corrupted.\n"
            << "Cannot read end of InitialSeeds when restoring random engine states.\n";
      }

      is >> leading >> stateVectorSize >> trailing;
      if (!is.good() || leading != std::string("<FullStateLength>") || trailing != std::string("</FullStateLength>")) {
        throw Exception(errors::Configuration)
            << "File \"" << restoreFileName_ << "\" is ill-structured or otherwise corrupted.\n"
            << "Cannot read state vector length when restoring random engine states.\n";
      }

      is >> leading;
      if (!is.good() || leading != std::string("<FullState>")) {
        throw Exception(errors::Configuration)
            << "File \"" << restoreFileName_ << "\" is ill-structured or otherwise corrupted.\n"
            << "Cannot read beginning of FullState when restoring random engine states.\n";
      }

      if (stateVectorSize > maxStates) {
        throw Exception(errors::Configuration)
            << "File \"" << restoreFileName_ << "\" is ill-structured or otherwise corrupted.\n"
            << "The number of states exceeds 64K.\n";
      }

      readVector(is, stateVectorSize, stateVector);

      is >> trailing;
      if (!is.good() || trailing != std::string("</FullState>")) {
        throw Exception(errors::Configuration)
            << "File \"" << restoreFileName_ << "\" is ill-structured or otherwise corrupted.\n"
            << "Cannot read end of FullState when restoring random engine states.\n";
      }

      if (saveToCache) {
        RandomEngineState randomEngineState;
        randomEngineState.setLabel(moduleLabel);
        std::vector<RandomEngineState>::iterator state =
            std::lower_bound(cache.begin(), cache.end(), randomEngineState);

        if (state != cache.end() && moduleLabel == state->getLabel()) {
          if (seedVector.size() != state->getSeed().size() || stateVector.size() != state->getState().size()) {
            throw Exception(errors::Configuration)
                << "File \"" << restoreFileName_ << "\" is ill-structured or otherwise corrupted.\n"
                << "Vectors containing engine state are the incorrect size for the type of random engine.\n";
          }
          state->setSeed(seedVector);
          state->setState(stateVector);
        }
      }
      return true;
    }

    void RandomNumberGeneratorService::readVector(std::istream& is, unsigned numItems, std::vector<std::uint32_t>& v) {
      v.clear();
      v.reserve(numItems);
      std::uint32_t data;
      for (unsigned i = 0; i < numItems; ++i) {
        is >> data;
        if (!is.good()) {
          throw Exception(errors::Configuration)
              << "File \"" << restoreFileName_ << "\" is ill-structured or otherwise corrupted.\n"
              << "Cannot read vector when restoring random engine states.\n";
        }
        v.push_back(data);
      }
    }

    void RandomNumberGeneratorService::createEnginesInVector(std::vector<LabelAndEngine>& engines,
                                                             unsigned int seedOffset,
                                                             unsigned int eventSeedOffset,
                                                             std::vector<ModuleIDToEngine>& moduleIDVector) {
      // The vectors we will fill here will be the same size as
      // or smaller than seedsAndNameMap_.
      engines.reserve(seedsAndNameMap_.size());
      moduleIDVector.reserve(seedsAndNameMap_.size());

      for (auto const& i : seedsAndNameMap_) {
        unsigned int moduleID = i.second.moduleID();
        if (moduleID != std::numeric_limits<unsigned int>::max()) {
          std::string const& label = i.first;
          std::string const& name = i.second.engineName();
          VUint32 const& seeds = i.second.seeds();

          if (name == "RanecuEngine") {
            std::shared_ptr<CLHEP::HepRandomEngine> engine = std::make_shared<CLHEP::RanecuEngine>();
            engines.emplace_back(label, seeds, engine);
            resetEngineSeeds(engines.back(), name, seeds, seedOffset, eventSeedOffset);
          }
          // For the other engines, one seed is required
          else {
            long int seedL = static_cast<long int>(seeds[0]);

            if (name == "HepJamesRandom") {
              std::shared_ptr<CLHEP::HepRandomEngine> engine = std::make_shared<CLHEP::HepJamesRandom>(seedL);
              engines.emplace_back(label, seeds, engine);
              if (seedOffset != 0 || eventSeedOffset != 0) {
                resetEngineSeeds(engines.back(), name, seeds, seedOffset, eventSeedOffset);
              }
            } else if (name == "MixMaxRng") {
              std::shared_ptr<CLHEP::HepRandomEngine> engine = std::make_shared<CLHEP::MixMaxRng>(seedL);
              engines.emplace_back(label, seeds, engine);
              if (seedOffset != 0 || eventSeedOffset != 0) {
                resetEngineSeeds(engines.back(), name, seeds, seedOffset, eventSeedOffset);
              }
            } else {  // TRandom3, currently the only other possibility

              // There is a dangerous conversion from std::uint32_t to long
              // that occurs above. In the next 2 lines we check the
              // behavior is what we need for the service to work
              // properly.  This conversion is forced on us by the
              // CLHEP and ROOT interfaces. If the assert ever starts
              // to fail we will have to come up with a way to deal
              // with this.
              std::uint32_t seedu32 = static_cast<std::uint32_t>(seedL);
              assert(seeds[0] == seedu32);

              std::shared_ptr<CLHEP::HepRandomEngine> engine = std::make_shared<TRandomAdaptor>(seedL);
              engines.emplace_back(label, seeds, engine);
              if (seedOffset != 0 || eventSeedOffset != 0) {
                resetEngineSeeds(engines.back(), name, seeds, seedOffset, eventSeedOffset);
              }
            }
          }
          moduleIDVector.emplace_back(&engines.back(), moduleID);
        }  // if moduleID valid
      }    // loop over seedsAndMap
      std::sort(moduleIDVector.begin(), moduleIDVector.end());
    }

    void RandomNumberGeneratorService::resetEngineSeeds(LabelAndEngine& labelAndEngine,
                                                        std::string const& engineName,
                                                        VUint32 const& seeds,
                                                        std::uint32_t offset1,
                                                        std::uint32_t offset2) {
      if (engineName == "RanecuEngine") {
        assert(seeds.size() == 2U);
        // Wrap around if the offsets push the seed over the maximum allowed value
        std::uint32_t mod = maxSeedRanecu + 1U;
        offset1 %= mod;
        offset2 %= mod;
        std::uint32_t seed0 = (seeds[0] + offset1) % mod;
        seed0 = (seed0 + offset2) % mod;
        labelAndEngine.setSeed(seed0, 0);
        labelAndEngine.setSeed(seeds[1], 1);
        long int seedL[2];
        seedL[0] = static_cast<long int>(seed0);
        seedL[1] = static_cast<long int>(seeds[1]);
        labelAndEngine.engine()->setSeeds(seedL, 0);
      } else {
        assert(seeds.size() == 1U);

        if (engineName == "HepJamesRandom" || engineName == "MixMaxRng") {
          // Wrap around if the offsets push the seed over the maximum allowed value
          std::uint32_t mod = maxSeedHepJames + 1U;
          offset1 %= mod;
          offset2 %= mod;
          std::uint32_t seed0 = (seeds[0] + offset1) % mod;
          seed0 = (seed0 + offset2) % mod;
          labelAndEngine.setSeed(seed0, 0);

          long int seedL = static_cast<long int>(seed0);
          labelAndEngine.engine()->setSeed(seedL, 0);
        } else {
          assert(engineName == "TRandom3");
          // Wrap around if the offsets push the seed over the maximum allowed value
          // We have to be extra careful with this one because it may also go beyond
          // the values 32 bits can hold
          std::uint32_t max32 = maxSeedTRandom3;
          std::uint32_t seed0 = seeds[0];
          if ((max32 - seed0) >= offset1) {
            seed0 += offset1;
          } else {
            seed0 = offset1 - (max32 - seed0) - 1U;
          }
          if ((max32 - seed0) >= offset2) {
            seed0 += offset2;
          } else {
            seed0 = offset2 - (max32 - seed0) - 1U;
          }
          labelAndEngine.setSeed(seed0, 0);

          long seedL = static_cast<long>(seed0);

          // There is a dangerous conversion from std::uint32_t to long
          // that occurs above. In the next 2 lines we check the
          // behavior is what we need for the service to work
          // properly.  This conversion is forced on us by the
          // CLHEP and ROOT interfaces. If the assert ever starts
          // to fail we will have to come up with a way to deal
          // with this.
          std::uint32_t seedu32 = static_cast<std::uint32_t>(seedL);
          assert(seed0 == seedu32);

          labelAndEngine.engine()->setSeed(seedL, 0);
        }
      }
    }
  }  // namespace service
}  // namespace edm
