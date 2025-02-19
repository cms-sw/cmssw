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

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterWildcard.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "IOMC/RandomEngine/src/TRandomAdaptor.h"
#include "SimDataFormats/RandomEngine/interface/RandomEngineState.h"
#include "SimDataFormats/RandomEngine/interface/RandomEngineStates.h"

#include "CLHEP/Random/engineIDulong.h"
#include "CLHEP/Random/JamesRandom.h"
#include "CLHEP/Random/RanecuEngine.h"

#include <iostream>
#include <limits>
#include <sstream>
#include <unistd.h>

namespace edm {
  namespace service {

    uint32_t RandomNumberGeneratorService::maxSeedRanecu =   2147483647U;
    uint32_t RandomNumberGeneratorService::maxSeedHepJames =  900000000U;
    uint32_t RandomNumberGeneratorService::maxSeedTRandom3 = 4294967295U;

    RandomNumberGeneratorService::RandomNumberGeneratorService(ParameterSet const& pset,
                                                               ActivityRegistry& activityRegistry):
      saveFileName_(pset.getUntrackedParameter<std::string>("saveFileName")),
      saveFileNameRecorded_(false),
      restoreFileName_(pset.getUntrackedParameter<std::string>("restoreFileName")),
      enableChecking_(pset.getUntrackedParameter<bool>("enableChecking")),
      firstLumi_(true),
      childIndex_(0U),
      eventSeedOffset_(pset.getUntrackedParameter<unsigned>("eventSeedOffset")),
      failedToFindStatesInLumi_(false) {

      if(pset.exists("restoreStateTag")) {
        restoreStateTag_ = pset.getUntrackedParameter<edm::InputTag>("restoreStateTag");
      } else {
        restoreStateTag_ = edm::InputTag(pset.getUntrackedParameter<std::string>("restoreStateLabel"), "", "");
      }
      restoreStateBeginLumiTag_ = edm::InputTag(restoreStateTag_.label(), "beginLumi", restoreStateTag_.process()); 

      if(!restoreFileName_.empty() && !restoreStateTag_.label().empty()) {
        throw Exception(errors::Configuration)
          << "In the configuration for the RandomNumberGeneratorService both\n"
          << "restoreFileName and restoreStateLabel were set to nonempty values\n"
          << "which is illegal.  It is impossible to restore the random engine\n"
          << "states two different ways in the same process.\n";
      }

      // The saveFileName must correspond to a file name without any path specification.
      // Throw if that is not true.
      if(!saveFileName_.empty() && (saveFileName_.find("/") != std::string::npos)) {
        throw Exception(errors::Configuration)
          << "The saveFileName parameter must be a simple file name with no path\n"
          << "specification. In the configuration, it was given the value \""
          << saveFileName_ << "\"\n";
      }

      // Check if the configuration file is still expressed in the old style.
      // We do this by looking for a PSet named moduleSeeds.  This parameter
      // is unique to an old style cfg file.
      if(pset.exists("moduleSeeds")) {
        oldStyleConfig(pset);
      } else {

        uint32_t initialSeed;
        VUint32 initialSeedSet;
        std::string engineName;

        VString pSets = pset.getParameterNamesForType<ParameterSet>();
        for(VString::const_iterator it = pSets.begin(), itEnd = pSets.end(); it != itEnd; ++it) {

          ParameterSet const& modulePSet = pset.getParameterSet(*it);
          engineName = modulePSet.getUntrackedParameter<std::string>("engineName", std::string("HepJamesRandom"));

          bool initialSeedExists = modulePSet.exists("initialSeed");
          bool initialSeedSetExists = modulePSet.exists("initialSeedSet");

          if(initialSeedExists && initialSeedSetExists) {
            throw Exception(errors::Configuration)
              << "For the module with the label \"" << *it << "\",\n"
              << "both the parameters \"initialSeed\" and \"initialSeedSet\"\n"
              << "have been set in the configuration. You must set one or\n"
              << "the other.  It is illegal to set both.\n";
          } else if(!initialSeedExists && !initialSeedSetExists) {
            throw Exception(errors::Configuration)
              << "For the module with the label \"" << *it << "\",\n"
              << "neither the parameter \"initialSeed\" nor \"initialSeedSet\"\n"
              << "has been set in the configuration. You must set one or\n"
              << "the other.\n";
          } else if(initialSeedExists) {
            initialSeed = modulePSet.getUntrackedParameter<uint32_t>("initialSeed");
            initialSeedSet.clear();
            initialSeedSet.push_back(initialSeed);
          } else if(initialSeedSetExists) {
            initialSeedSet = modulePSet.getUntrackedParameter<VUint32>("initialSeedSet");
          }
          seedMap_[*it] = initialSeedSet;
          engineNameMap_[*it] = engineName;

          // For the CLHEP::RanecuEngine case, require a seed set containing exactly two seeds.

          if(engineName == std::string("RanecuEngine")) {
            if(initialSeedSet.size() != 2U) {
              throw Exception(errors::Configuration)
                << "Random engines of type \"RanecuEngine\" require 2 seeds\n"
                << "be specified with the parameter named \"initialSeedSet\".\n"
                << "Either \"initialSeedSet\" was not in the configuration\n"
                << "or its size was not 2 for the module with label \"" << *it << "\".\n" ;
            }
            boost::shared_ptr<CLHEP::HepRandomEngine> engine(new CLHEP::RanecuEngine());
            engineMap_[*it] = engine;

            if(initialSeedSet[0] > maxSeedRanecu ||
                initialSeedSet[1] > maxSeedRanecu) {  // They need to fit in a 31 bit integer
              throw Exception(errors::Configuration)
                << "The RanecuEngine seeds should be in the range 0 to 2147483647.\n"
                << "The seeds passed to the RandomNumberGenerationService from the\n"
                   "configuration file were " << initialSeedSet[0] << " and " << initialSeedSet[1]
                << "\nThis was for the module with label \"" << *it << "\".\n";
            }
            long int seedL[2];
            seedL[0] = static_cast<long int>(initialSeedSet[0]);
            seedL[1] = static_cast<long int>(initialSeedSet[1]);
            engine->setSeeds(seedL, 0);
          }
          // For the other engines, one seed is required
          else {
            if(initialSeedSet.size() != 1U) {
              throw Exception(errors::Configuration)
                << "Random engines of type \"HepJamesRandom\" and \"TRandom3\n"
                << "require exactly 1 seed be specified in the configuration.\n"
                << "There were " << initialSeedSet.size() << " seeds set for the\n"
                << "module with label \"" << *it << "\".\n" ;
            }
            long int seedL = static_cast<long int>(initialSeedSet[0]);

            if(engineName == "HepJamesRandom") {
              if(initialSeedSet[0] > maxSeedHepJames) {
                throw Exception(errors::Configuration)
                  << "The CLHEP::HepJamesRandom engine seed should be in the range 0 to 900000000.\n"
                  << "The seed passed to the RandomNumberGenerationService from the\n"
                     "configuration file was " << initialSeedSet[0] << ".  This was for \n"
                  << "the module with label " << *it << ".\n";
              }
              boost::shared_ptr<CLHEP::HepRandomEngine> engine(new CLHEP::HepJamesRandom(seedL));
              engineMap_[*it] = engine;
            } else if(engineName == "TRandom3") {

              // There is a dangerous conversion from uint32_t to long
              // that occurs above. In the next 2 lines we check the
              // behavior is what we need for the service to work
              // properly.  This conversion is forced on us by the
              // CLHEP and ROOT interfaces. If the assert ever starts
              // to fail we will have to come up with a way to deal
              // with this.
              uint32_t seedu32 = static_cast<uint32_t>(seedL);
              assert(initialSeedSet[0] == seedu32);

              boost::shared_ptr<CLHEP::HepRandomEngine> engine(new TRandomAdaptor(seedL));
              engineMap_[*it] = engine;
            } else {
              throw Exception(errors::Configuration)
                << "The random engine name, \"" << engineName
                << "\", does not correspond to a supported engine.\n"
                << "This engine was configured for the module with label \"" << *it << "\"";
            }
          }
        }
      }

      activityRegistry.watchPostBeginLumi(this, &RandomNumberGeneratorService::postBeginLumi);

      activityRegistry.watchPreModuleConstruction(this, &RandomNumberGeneratorService::preModuleConstruction);
      activityRegistry.watchPostModuleConstruction(this, &RandomNumberGeneratorService::postModuleConstruction);

      activityRegistry.watchPreModuleBeginJob(this, &RandomNumberGeneratorService::preModuleBeginJob);
      activityRegistry.watchPostModuleBeginJob(this, &RandomNumberGeneratorService::postModuleBeginJob);

      activityRegistry.watchPreModuleBeginRun(this, &RandomNumberGeneratorService::preModuleBeginRun);
      activityRegistry.watchPostModuleBeginRun(this, &RandomNumberGeneratorService::postModuleBeginRun);

      activityRegistry.watchPreModuleBeginLumi(this, &RandomNumberGeneratorService::preModuleBeginLumi);
      activityRegistry.watchPostModuleBeginLumi(this, &RandomNumberGeneratorService::postModuleBeginLumi);

      activityRegistry.watchPreModule(this, &RandomNumberGeneratorService::preModule);
      activityRegistry.watchPostModule(this, &RandomNumberGeneratorService::postModule);

      activityRegistry.watchPreModuleEndLumi(this, &RandomNumberGeneratorService::preModuleEndLumi);
      activityRegistry.watchPostModuleEndLumi(this, &RandomNumberGeneratorService::postModuleEndLumi);

      activityRegistry.watchPreModuleEndRun(this, &RandomNumberGeneratorService::preModuleEndRun);
      activityRegistry.watchPostModuleEndRun(this, &RandomNumberGeneratorService::postModuleEndRun);

      activityRegistry.watchPreModuleEndJob(this, &RandomNumberGeneratorService::preModuleEndJob);
      activityRegistry.watchPostModuleEndJob(this, &RandomNumberGeneratorService::postModuleEndJob);

      activityRegistry.watchPostForkReacquireResources(this, &RandomNumberGeneratorService::postForkReacquireResources);

      // the default for the stack is to point to the 'end' of our map which is used to define not set
      engineStack_.push_back(engineMap_.end());
      currentEngine_ = engineMap_.end();

      labelStack_.push_back(std::string());
      currentLabel_ = std::string();
    }

    RandomNumberGeneratorService::~RandomNumberGeneratorService() {
    }

    CLHEP::HepRandomEngine&
    RandomNumberGeneratorService::getEngine() const {

      if(currentEngine_ == engineMap_.end()) {
        if(currentLabel_ != std::string()) {
          throw Exception(errors::Configuration)
            << "The module with label \""
            << currentLabel_
            << "\" requested a random number engine from the \n"
               "RandomNumberGeneratorService, but that module was not configured\n"
               "for random numbers.  An engine is created only if a seed(s) is provided\n"
               "in the configuration file.  Please add the following PSet to the\n"
               "configuration file for the RandomNumberGeneratorService:\n\n"
               "  " << currentLabel_ << " = cms.PSet(\n"
               "    initialSeed = cms.untracked.uint32(your_seed),\n"
               "    engineName = cms.untracked.string('TRandom3')\n"
               "  )\n"
               "where you replace \"your_seed\" with a number and add a comma if necessary\n"
              "The \"engineName\" parameter is optional. If absent the default is \"HepJamesRandom\".\n";
        } else {
          throw Exception(errors::Unknown)
             << "Requested a random number engine from the RandomNumberGeneratorService\n"
                "when no module was active.  This is not supposed to be possible.\n"
                "Please inform the edm developers about this. It would be helpful to\n"
                "know the stack. If a source was requesting a random engine this could\n"
                "happen. Sources are not supposed to be doing that anymore.\n";
        }
      }
      return *(currentEngine_->second);
    }

    uint32_t
    RandomNumberGeneratorService::mySeed() const {

      std::map<std::string, VUint32>::const_iterator iter;
      iter = seedMap_.find(currentLabel_);

      if(iter == seedMap_.end()) {
        if(currentLabel_ != std::string()) {
          throw Exception(errors::Configuration)
            << "The module with label \""
            << currentLabel_
            << "\" requested a random number seed from the \n"
               "RandomNumberGeneratorService, but that module was not configured\n"
               "for random numbers.  An engine is created only if a seed(s) is provided\n"
               "in the configuration file.  Please add the following PSet to the\n"
               "configuration file for the RandomNumberGeneratorService:\n\n"
               "  " << currentLabel_ << " = cms.PSet(\n"
               "    initialSeed = cms.untracked.uint32(your_seed),\n"
               "    engineName = cms.untracked.string('TRandom3')\n"
               "  )\n"
               "where you replace \"your_seed\" with a number and add a comma if necessary\n"
              "The \"engineName\" parameter is optional. If absent the default is \"HepJamesRandom\".\n";
        } else {
          throw Exception(errors::Unknown)
             << "Requested a random number seed from the RandomNumberGeneratorService\n"
                "when no module was active.  This is not supposed to be possible.\n"
                "Please inform the edm developers about this. It would be helpful to\n"
                "know the stack. If a source was requesting a random engine this could\n"
                "happen. Sources are not supposed to be doing that anymore.\n";
        }
      }
      return iter->second[0];
    }

    void
    RandomNumberGeneratorService::fillDescriptions(ConfigurationDescriptions& descriptions) {
      ParameterSetDescription desc;

      std::string emptyString;
      edm::InputTag emptyInputTag("", "", "");

      desc.addNode( edm::ParameterDescription<edm::InputTag>("restoreStateTag", emptyInputTag, false) xor
                    edm::ParameterDescription<std::string>("restoreStateLabel", emptyString, false) );

      desc.addUntracked<std::string>("saveFileName", emptyString);
      desc.addUntracked<std::string>("restoreFileName", emptyString);
      desc.addUntracked<bool>("enableChecking", false);
      desc.addUntracked<unsigned>("eventSeedOffset", 0U);

      ParameterSetDescription val;
      // When the migration away from the deprecated interface is complete it would be better
      // to change the next line to a declaration of a single parameter named initialSeed instead
      // of being a wildcard.  Also the next two lines might also be combined with an "exclusive or"
      // operator.
      val.addWildcardUntracked<uint32_t>("*")->setComment("In the new interface, this wildcard will "
        "match either nothing or one parameter named initialSeed.  Either initialSeed will exist or "
        "initialSeedSet will exist but not both.  In the old deprecated interface, this will match "
        "parameters with the names being the module labels and the values being the seeds");
      val.addOptionalUntracked<std::vector<uint32_t> >("initialSeedSet")->setComment("New interface only");
      val.addOptionalUntracked<std::string>("engineName",std::string("HepJamesRandom"))->setComment("New interface only");

      ParameterWildcard<ParameterSetDescription> wnode("*", RequireZeroOrMore, true, val);
      wnode.setComment("In the new interface, the name of each ParameterSet will be the associated module label. "
                       "In the old deprecated interface there will be one ParameterSet named moduleSeeds");
      desc.addNode(wnode);

      // This only exists for backward compatibility reasons
      // This should be removed if all the configurations are
      // ever upgraded properly.
      desc.addOptionalUntracked<uint32_t>("sourceSeed")->
        setComment("This parameter is deprecated, has no effect and will likely be completely removed someday");

      descriptions.add("RandomNumberGeneratorService", desc);
    }

    void
    RandomNumberGeneratorService::postForkReacquireResources(unsigned childIndex, unsigned /*kMaxChildren*/) {
      childIndex_ = childIndex;

      if(!saveFileName_.empty()) {
        std::ostringstream suffix;
        suffix << "_" << childIndex;
        saveFileName_ += suffix.str();
      }
    }

    // The next three functions contain the complex logic
    // such that things occur in the proper sequence to be
    // able to save and restore the states.

    void
    RandomNumberGeneratorService::preBeginLumi(LuminosityBlock const& lumi) {

      if(firstLumi_) {
        // copy state from engines to lumi cache
        snapShot(lumiCache_);

        if(!restoreFileName_.empty()) {
          // copy state from text file to lumi cache
          readLumiStatesFromTextFile(restoreFileName_);
        }
      } else {
        snapShot(eventCache_);
      }

      // copy state from LuminosityBlock to lumi cache
      if(!restoreStateTag_.label().empty()) {
        readFromLuminosityBlock(lumi);
      }

      if(!firstLumi_ || !restoreFileName_.empty() || !restoreStateTag_.label().empty()) {
        // copy state from lumi cache to engines
        restoreFromCache(lumiCache_);
      }
    }

    // During the beginLumi processing the producer will copy the
    // the lumi cache to a product if the producer was scheduled
    // in a path in the configuration

    void
    RandomNumberGeneratorService::postBeginLumi(LuminosityBlock const&, EventSetup const&) {

      if(firstLumi_) {
        // reset state with new seeds based on child index
        startNewSequencesForEvents();
        if(!restoreFileName_.empty()) {
          snapShot(eventCache_);
          // copy state from text file to event cache
          readEventStatesFromTextFile(restoreFileName_);
        }
      }
      if(!firstLumi_ || !restoreFileName_.empty()) {
        // copy state from event cache to engines
        restoreFromCache(eventCache_);
      }
      firstLumi_ = false;
    }

    void
    RandomNumberGeneratorService::postEventRead(Event const& event) {
      // copy from Event to event cache
      if(!restoreStateTag_.label().empty()) {
        snapShot(eventCache_);
        readFromEvent(event);

        // copy from event cache to engines
        restoreFromCache(eventCache_);
      } else {
        // copy from engines to event cache
        snapShot(eventCache_);
      }
      // if requested write text file from both caches
      if(!saveFileName_.empty())  {
        saveStatesToFile(saveFileName_);
        if(!saveFileNameRecorded_) {
          std::string fullName = constructSaveFileName();
          Service<JobReport> reportSvc;
          reportSvc->reportRandomStateFile(fullName);
          saveFileNameRecorded_ = true;
        }
      }
    }

    // During the event processing the producer will copy the
    // the event cache to a product if the producer was scheduled
    // in a path in the configuration

    void
    RandomNumberGeneratorService::preModuleConstruction(ModuleDescription const& description) {
      push(description.moduleLabel());
      if(enableChecking_ && currentEngine_ != engineMap_.end()) {
        engineStateStack_.push_back(currentEngine_->second->put());
      }
    }

    void
    RandomNumberGeneratorService::postModuleConstruction(ModuleDescription const& description) {
      if(enableChecking_ && currentEngine_ != engineMap_.end()) {
        if(engineStateStack_.back() != currentEngine_->second->put()) {
          throw Exception(errors::LogicError)
            << "It is illegal to generate random numbers during module construction because \n"
               "that makes it very difficult to reproduce the processing of individual\n"
               "events.  Random numbers were generated during module construction for the module with\n"
               "class name \"" << description.moduleName() << "\"\n"
               "and module label \"" << description.moduleLabel() << "\"\n";
        }
        engineStateStack_.pop_back();
      }
      pop();
    }

    void
    RandomNumberGeneratorService::preModuleBeginJob(ModuleDescription const& description) {
      push(description.moduleLabel());
      if(enableChecking_ && currentEngine_ != engineMap_.end()) {
        engineStateStack_.push_back(currentEngine_->second->put());
      }
    }

    void
    RandomNumberGeneratorService::postModuleBeginJob(ModuleDescription const& description) {
      if(enableChecking_ && currentEngine_ != engineMap_.end()) {
        if(engineStateStack_.back() != currentEngine_->second->put()) {
          throw Exception(errors::LogicError)
            << "It is illegal to generate random numbers during beginJob because \n"
               "that makes it very difficult to reproduce the processing of individual\n"
               "events.  Random numbers were generated during beginJob for the module with\n"
               "class name \"" << description.moduleName() << "\"\n"
               "and module label \"" << description.moduleLabel() << "\"\n";
        }
        engineStateStack_.pop_back();
      }
      pop();
    }

    void
    RandomNumberGeneratorService::preModuleBeginRun(ModuleDescription const& description) {
      push(description.moduleLabel());
      if(enableChecking_ && currentEngine_ != engineMap_.end()) {
        engineStateStack_.push_back(currentEngine_->second->put());
      }
    }

    void
    RandomNumberGeneratorService::postModuleBeginRun(ModuleDescription const& description) {
      if(enableChecking_ && currentEngine_ != engineMap_.end()) {
        if(engineStateStack_.back() != currentEngine_->second->put()) {
          throw Exception(errors::LogicError)
            << "It is illegal to generate random numbers during beginRun because \n"
               "that makes it very difficult to reproduce the processing of individual\n"
               "events.  Random numbers were generated during beginRun for the module with\n"
               "class name \"" << description.moduleName() << "\"\n"
               "and module label \"" << description.moduleLabel() << "\"\n";
        }
        engineStateStack_.pop_back();
      }
      pop();
    }

    void
    RandomNumberGeneratorService::preModuleBeginLumi(ModuleDescription const& description) {
      push(description.moduleLabel());
    }

    void
    RandomNumberGeneratorService::postModuleBeginLumi(ModuleDescription const&) {
      pop();
    }

    void
    RandomNumberGeneratorService::preModule(ModuleDescription const& description) {
      push(description.moduleLabel());
    }

    void
    RandomNumberGeneratorService::postModule(ModuleDescription const&) {
      pop();
    }

    void
    RandomNumberGeneratorService::preModuleEndLumi(ModuleDescription const& description) {
      push(description.moduleLabel());
      if(enableChecking_ && currentEngine_ != engineMap_.end()) {
        engineStateStack_.push_back(currentEngine_->second->put());
      }
    }

    void
    RandomNumberGeneratorService::postModuleEndLumi(ModuleDescription const& description) {
      if(enableChecking_ && currentEngine_ != engineMap_.end()) {
        if(engineStateStack_.back() != currentEngine_->second->put()) {
          throw Exception(errors::LogicError)
            << "It is illegal to generate random numbers during endLumi because \n"
               "that makes it very difficult to reproduce the processing of individual\n"
               "events.  Random numbers were generated during endLumi for the module with\n"
               "class name \"" << description.moduleName() << "\"\n"
               "and module label \"" << description.moduleLabel() << "\"\n";
        }
        engineStateStack_.pop_back();
      }
      pop();
    }

    void
    RandomNumberGeneratorService::preModuleEndRun(ModuleDescription const& description) {
      push(description.moduleLabel());
      if(enableChecking_ && currentEngine_ != engineMap_.end()) {
        engineStateStack_.push_back(currentEngine_->second->put());
      }
    }

    void
    RandomNumberGeneratorService::postModuleEndRun(ModuleDescription const& description) {
      if(enableChecking_ && currentEngine_ != engineMap_.end()) {
        if(engineStateStack_.back() != currentEngine_->second->put()) {
          throw Exception(errors::LogicError)
            << "It is illegal to generate random numbers during endRun because \n"
               "that makes it very difficult to reproduce the processing of individual\n"
               "events.  Random numbers were generated during endRun for the module with\n"
               "class name \"" << description.moduleName() << "\"\n"
               "and module label \"" << description.moduleLabel() << "\"\n";
        }
        engineStateStack_.pop_back();
      }
      pop();
    }

    void
    RandomNumberGeneratorService::preModuleEndJob(ModuleDescription const& description) {
      push(description.moduleLabel());
      if(enableChecking_ && currentEngine_ != engineMap_.end()) {
        engineStateStack_.push_back(currentEngine_->second->put());
      }
    }

    void
    RandomNumberGeneratorService::postModuleEndJob(ModuleDescription const& description) {
      if(enableChecking_ && currentEngine_ != engineMap_.end()) {
        if(engineStateStack_.back() != currentEngine_->second->put()) {
          throw Exception(errors::LogicError)
            << "It is illegal to generate random numbers during endJob because \n"
               "that makes it very difficult to reproduce the processing of individual\n"
               "events.  Random numbers were generated during endJob for the module with\n"
               "class name \"" << description.moduleName() << "\"\n"
               "and module label \"" << description.moduleLabel() << "\"\n";
        }
        engineStateStack_.pop_back();
      }
      pop();
    }

    std::vector<RandomEngineState> const&
    RandomNumberGeneratorService::getLumiCache() const {
      return lumiCache_;
    }

    std::vector<RandomEngineState> const&
    RandomNumberGeneratorService::getEventCache() const {
      return eventCache_;
    }

    void
    RandomNumberGeneratorService::print() {
      std::cout << "\n\nRandomNumberGeneratorService dump\n\n";

      std::cout << "    Contents of seedMap\n";
      for(std::map<std::string, std::vector<uint32_t> >::const_iterator iter = seedMap_.begin();
           iter != seedMap_.end();
           ++iter) {
        std::cout << "        " << iter->first;
        std::vector<uint32_t> seeds = iter->second;
        for(std::vector<uint32_t>::const_iterator vIter = seeds.begin();
             vIter != seeds.end();
             ++vIter) {
          std::cout << "   "  << *vIter;
        }
        std::cout << "\n";
      }
      std::cout << "\n    Contents of engineNameMap\n";
      for(std::map<std::string, std::string>::const_iterator iter = engineNameMap_.begin();
           iter != engineNameMap_.end();
           ++iter) {
        std::cout << "        " << iter->first << "   " << iter->second << "\n";
      }
      std::cout << "\n    Contents of engineMap\n";
      for(EngineMap::const_iterator iter = engineMap_.begin();
           iter != engineMap_.end();
           ++iter) {
        std::cout << "        " << iter->first
                  << "   " << iter->second->name() << "   ";
        if(iter->second->name() == std::string("HepJamesRandom")) {
          std::cout << iter->second->getSeed();
        } else {
          std::cout << "Engine does not know original seed";
        }
        std::cout << "\n";
      }
      std::cout << "\n";
      std::cout << "    currentLabel_ = " << currentLabel_ << "\n";
      std::cout << "    labelStack_ size = " << labelStack_.size() << "\n";
      int i = 0;
      for(VString::const_iterator iter = labelStack_.begin();
           iter != labelStack_.end();
           ++iter, ++i) {
        std::cout << "                 " << i << "  " << *iter << "\n";
      }
      if(currentEngine_ == engineMap_.end()) {
        std::cout << "    currentEngine points to end\n";
      } else {
        std::cout << "    currentEngine_ = " << currentEngine_->first
                  << "  " << currentEngine_->second->name()
                  << "  " << currentEngine_->second->getSeed() << "\n";
      }

      std::cout << "    engineStack_ size = " << engineStack_.size() << "\n";
      i = 0;
      for(std::vector<EngineMap::const_iterator>::const_iterator iter = engineStack_.begin();
           iter != engineStack_.end();
           ++iter, ++i) {
        if(*iter == engineMap_.end()) {
          std::cout << "                 " << i << "  Points to end of engine map\n";
        } else {
          std::cout << "                 " << i << "  " << (*iter)->first
                    << "  " << (*iter)->second->name() << "  " << (*iter)->second->getSeed() << "\n";
        }
      }

      std::cout << "    restoreStateTag_ = " << restoreStateTag_ << "\n";
      std::cout << "    saveFileName_ = " << saveFileName_ << "\n";
      std::cout << "    restoreFileName_ = " << restoreFileName_ << "\n";
    }

    void
    RandomNumberGeneratorService::push(std::string const& iLabel) {
      currentEngine_ = engineMap_.find(iLabel);
      engineStack_.push_back(currentEngine_);

      labelStack_.push_back(iLabel);
      currentLabel_ = iLabel;
    }

    void
    RandomNumberGeneratorService::pop() {
      engineStack_.pop_back();
      //NOTE: algorithm is such that we always have at least one item in the stacks
      currentEngine_ = engineStack_.back();
      labelStack_.pop_back();
      currentLabel_ = labelStack_.back();
    }

    void
    RandomNumberGeneratorService::readFromLuminosityBlock(LuminosityBlock const& lumi) {

      Handle<RandomEngineStates> states;
      lumi.getByLabel(restoreStateBeginLumiTag_, states);

      if(!states.isValid()) {
        failedToFindStatesInLumi_ = true;
        return;
      }
      failedToFindStatesInLumi_ = false;
      states->getRandomEngineStates(lumiCache_);
    }

    void
    RandomNumberGeneratorService::readFromEvent(Event const& event) {

      Handle<RandomEngineStates> states;

      event.getByLabel(restoreStateTag_, states);

      if(!states.isValid()) {
        if(failedToFindStatesInLumi_ && backwardCompatibilityRead(event)) {
          return;
        } else {
          throw Exception(errors::ProductNotFound)
            << "The RandomNumberGeneratorService is trying to restore\n"
            << "the state of the random engines by reading a product from\n"
            << "the Event with input tag \"" << restoreStateTag_ << "\".  It\n"
            << "fails to find one.  The label used in the request for the product\n"
            << "is set in the configuration. It is probably set to the wrong value\n"
            << "in the configuration file.  It must match the module label\n"
            << "of the RandomEngineStateProducer that created the product in\n"
            << "a previous process\n";
        }
      }
      if(failedToFindStatesInLumi_) {
        throw Exception(errors::ProductNotFound)
          << "The RandomNumberGeneratorService is trying to restore\n"
          << "the state of the random engines by reading a product from\n"
          << "the Event and LuminosityBlock with input tag \"" << restoreStateTag_ << "\".\n"
          << "It found the product in the Event but not the one in the LuminosityBlock.\n"
          << "Either the product in the LuminosityBlock was dropped or\n"
          << "there is a bug somewhere\n";
      }
      states->getRandomEngineStates(eventCache_);
    }

    bool
    RandomNumberGeneratorService::backwardCompatibilityRead(Event const& event) {

      Handle<std::vector<RandomEngineState> > states;

      event.getByLabel(restoreStateTag_, states);
      if(!states.isValid()) {
        return false;
      }
      for(std::vector<RandomEngineState>::const_iterator state = states->begin(),
                                                          iEnd = states->end();
           state != iEnd; ++state) {

        std::vector<RandomEngineState>::iterator cachedState =
          std::lower_bound(eventCache_.begin(), eventCache_.end(), *state);


        if(cachedState != eventCache_.end() && cachedState->getLabel() == state->getLabel()) {
          if(cachedState->getSeed().size() != state->getSeed().size() ||
              cachedState->getState().size() != state->getState().size()) {
            throw Exception(errors::Configuration)
              << "In function RandomNumberGeneratorService::backwardCompatibilityRead.\n"
              << "When attempting to replay processing with the RandomNumberGeneratorService,\n"
              << "the engine type for each module must be the same in the replay configuration\n"
              << "and the original configuration.  If this is not the problem, then the data\n"
              << "is somehow corrupted or there is a bug because the vector in the data containing\n"
              << "the seeds or engine state is the incorrect size for the type of random engine.\n";
          }
          cachedState->setSeed(state->getSeed());
          cachedState->setState(state->getState());
        }
      }
      return true;
    }

    void
    RandomNumberGeneratorService::snapShot(std::vector<RandomEngineState>& cache) {
      cache.resize(engineMap_.size());
      std::vector<RandomEngineState>::iterator state = cache.begin();

      for(EngineMap::const_iterator iter = engineMap_.begin();
           iter != engineMap_.end();
           ++iter, ++state) {

        state->setLabel(iter->first);
        state->setSeed(seedMap_[iter->first]);

        std::vector<unsigned long> stateL = iter->second->put();
        state->clearStateVector();
        state->reserveStateVector(stateL.size());
        for(std::vector<unsigned long>::const_iterator vIter = stateL.begin();
             vIter != stateL.end();
             ++vIter) {
          state->push_back_stateVector(static_cast<uint32_t>(*vIter));
        }
      }
    }

    void
    RandomNumberGeneratorService::restoreFromCache(std::vector<RandomEngineState> const& cache) {
      for(std::vector<RandomEngineState>::const_iterator iter = cache.begin(),
                                                          iEnd = cache.end();
           iter != iEnd; ++iter) {

        std::string const& engineLabel = iter->getLabel();

        std::vector<uint32_t> const& engineState = iter->getState();
        std::vector<unsigned long> engineStateL;
        for(std::vector<uint32_t>::const_iterator iVal = engineState.begin(),
                                                 theEnd = engineState.end();
             iVal != theEnd; ++iVal) {
          engineStateL.push_back(static_cast<unsigned long>(*iVal));
        }

        std::vector<uint32_t> const& engineSeeds = iter->getSeed();
        std::vector<long> engineSeedsL;
        for(std::vector<uint32_t>::const_iterator iVal = engineSeeds.begin(),
                                                 theEnd = engineSeeds.end();
          iVal != theEnd;
          ++iVal) {
          long seedL = static_cast<long>(*iVal);
          engineSeedsL.push_back(seedL);

          // There is a dangerous conversion from uint32_t to long
          // that occurs above. In the next 2 lines we check the
          // behavior is what we need for the service to work
          // properly.  This conversion is forced on us by the
          // CLHEP and ROOT interfaces. If the assert ever starts
          // to fail we will have to come up with a way to deal
          // with this.
          uint32_t seedu32 = static_cast<uint32_t>(seedL);
          assert(*iVal == seedu32);
        }

        EngineMap::iterator engine = engineMap_.find(engineLabel);

        if(engine != engineMap_.end()) {

          seedMap_[engineLabel] = engineSeeds;

          // We need to handle each type of engine differently because each
          // has different requirements on the seed or seeds.
          if(engineStateL[0] == CLHEP::engineIDulong<CLHEP::HepJamesRandom>()) {

            checkEngineType(engine->second->name(), std::string("HepJamesRandom"), engineLabel);

            // These two lines actually restore the seed and engine state.
            engine->second->setSeed(engineSeedsL[0], 0);
            engine->second->get(engineStateL);
          } else if(engineStateL[0] == CLHEP::engineIDulong<CLHEP::RanecuEngine>()) {

            checkEngineType(engine->second->name(), std::string("RanecuEngine"), engineLabel);

            // This line actually restores the engine state.
            engine->second->get(engineStateL);
          } else if(engineStateL[0] == CLHEP::engineIDulong<TRandomAdaptor>()) {

            checkEngineType(engine->second->name(), std::string("TRandom3"), engineLabel);

            // This line actually restores the engine state.
            engine->second->setSeed(engineSeedsL[0], 0);
            engine->second->get(engineStateL);
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
        }
      }
    }

    void
    RandomNumberGeneratorService::checkEngineType(std::string const& typeFromConfig,
                                                  std::string const& typeFromEvent,
                                                  std::string const& engineLabel) {
      if(typeFromConfig != typeFromEvent) {
        throw Exception(errors::Configuration)
          << "The RandomNumberGeneratorService is trying to restore\n"
          << "the state of the random engine for the module \""
          << engineLabel << "\".  An\n"
          << "error was detected because the type of the engine in the\n"
          << "input file and the configuration file do not match.\n"
          << "In the configuration file the type is \"" << typeFromConfig
          << "\".\nIn the input file the type is \"" << typeFromEvent << "\".  If\n"
          << "you are not generating any random numbers in this module, then\n"
          << "remove the line in the configuration file that gives it\n"
          << "a seed and the error will go away.  Otherwise, you must give\n"
          << "this module the same engine type in the configuration file or\n"
          << "stop trying to restore the random engine state.\n";
      }
    }

    void
    RandomNumberGeneratorService::saveStatesToFile(std::string const& fileName) {
      if(!outFile_.is_open()) {
        outFile_.open(fileName.c_str(), std::ofstream::out | std::ofstream::trunc);
      }
      if(!outFile_) {
        throw Exception(errors::Configuration)
          << "Unable to open the file \""
          << fileName << "\" to save the state of the random engines.\n";
      }
      outFile_.seekp(0, std::ios_base::beg);
      outFile_ << "<RandomEngineStates>\n";

      outFile_ << "<Event>\n";
      writeStates(eventCache_, outFile_);
      outFile_ << "</Event>\n" ;

      outFile_ << "<Lumi>\n";
      writeStates(lumiCache_, outFile_);
      outFile_ << "</Lumi>\n" ;

      outFile_ << "</RandomEngineStates>\n" ;
      outFile_.flush();
    }

    void
    RandomNumberGeneratorService::writeStates(std::vector<RandomEngineState> const& v,
                                              std::ofstream& outFile) {
      for(std::vector<RandomEngineState>::const_iterator iter = v.begin(),
                                                          iEnd = v.end();
        iter != iEnd; ++iter) {

        std::vector<uint32_t> const& seedVector = iter->getSeed();
        std::vector<uint32_t>::size_type seedVectorLength = seedVector.size();

        std::vector<uint32_t> const& stateVector = iter->getState();
        std::vector<uint32_t>::size_type stateVectorLength = stateVector.size();

        outFile << "<ModuleLabel>\n" << iter->getLabel() << "\n</ModuleLabel>\n";

        outFile << "<SeedLength>\n" << seedVectorLength << "\n</SeedLength>\n" ;
        outFile << "<InitialSeeds>\n";
        writeVector(seedVector, outFile);
        outFile << "</InitialSeeds>\n";
        outFile << "<FullStateLength>\n" << stateVectorLength << "\n</FullStateLength>\n";
        outFile << "<FullState>\n";
        writeVector(stateVector, outFile);
        outFile   << "</FullState>\n";
      }
    }

    void
    RandomNumberGeneratorService::writeVector(VUint32 const& v,
                                              std::ofstream& outFile) {
      if(v.empty()) return;
      size_t numItems = v.size();
      for(size_t i = 0; i < numItems; ++i)  {
        if(i != 0 && i % 10 == 0) outFile << "\n";
        outFile << std::setw(13) << v[i];
      }
      outFile << "\n";
    }

    std::string RandomNumberGeneratorService::constructSaveFileName() {
      char directory[1500];
      std::string fullName(getcwd(directory, sizeof(directory)) ? directory : "/PathIsTooBig");
      fullName += "/" + saveFileName_;
      return fullName;
    }

    void
    RandomNumberGeneratorService::readEventStatesFromTextFile(std::string const& fileName) {
      std::string whichStates("<Event>");
      readStatesFromFile(fileName, eventCache_, whichStates);
    }

    void
    RandomNumberGeneratorService::readLumiStatesFromTextFile(std::string const& fileName) {
      std::string whichStates("<Lumi>");
      readStatesFromFile(fileName, lumiCache_, whichStates);
    }


    void
    RandomNumberGeneratorService::readStatesFromFile(std::string const& fileName,
                                                     std::vector<RandomEngineState>& cache,
                                                     std::string const& whichStates) {
      std::ifstream inFile;
      inFile.open(fileName.c_str(), std::ifstream::in);
      if(!inFile) {
        throw Exception(errors::Configuration)
          << "Unable to open the file \""
          << fileName << "\" to restore the random engine states.\n";
      }

      std::string text;
      inFile >> text;
      if(!inFile.good() || text != std::string("<RandomEngineStates>")) {
        throw Exception(errors::Configuration)
          << "Attempting to read file with random number engine states.\n"
          << "File \"" << restoreFileName_
          << "\" is ill-structured or otherwise corrupted.\n"
          << "Cannot read the file header word.\n";
      }
      bool saveToCache = false;
      while(readEngineState(inFile, cache, whichStates, saveToCache)) {}
    }

    bool RandomNumberGeneratorService::readEngineState(std::istream& is,
                                                       std::vector<RandomEngineState>& cache,
                                                       std::string const& whichStates,
                                                       bool& saveToCache) {
      std::string leading;
      std::string trailing;
      std::string moduleLabel;
      std::vector<uint32_t>::size_type seedVectorSize;
      std::vector<uint32_t> seedVector;
      std::vector<uint32_t>::size_type stateVectorSize;
      std::vector<uint32_t> stateVector;

      // First we need to look for the special strings
      // that mark the end of the file and beginning and
      // and end of the data for different sections.

      is >> leading;
      if(!is.good()) {
        throw Exception(errors::Configuration)
          << "File \"" << restoreFileName_
          << "\" is ill-structured or otherwise corrupted.\n"
          << "Cannot read next field and did not hit the end yet.\n";
      }

      // This marks the end of the file. We are done.
      if(leading == std::string("</RandomEngineStates>")) return false;

      // This marks the end of a section of the data
      if(leading == std::string("</Event>") ||
          leading == std::string("</Lumi>")) {
        saveToCache = false;
        return true;
      }

      // This marks the beginning of a section
      if(leading == std::string("<Event>") ||
          leading == std::string("<Lumi>")) {
        saveToCache = (leading == whichStates);
        return true;
      }

      // Process the next engine state

      is >> moduleLabel >> trailing;
      if(!is.good() ||
          leading != std::string("<ModuleLabel>") ||
          trailing != std::string("</ModuleLabel>")) {
        throw Exception(errors::Configuration)
          << "File \"" << restoreFileName_
          << "\" is ill-structured or otherwise corrupted.\n"
          << "Cannot read a module label when restoring random engine states.\n";
      }

      is >> leading >> seedVectorSize >> trailing;
      if(!is.good() ||
          leading != std::string("<SeedLength>") ||
          trailing != std::string("</SeedLength>")) {
        throw Exception(errors::Configuration)
          << "File \"" << restoreFileName_
          << "\" is ill-structured or otherwise corrupted.\n"
          << "Cannot read seed vector length when restoring random engine states.\n";
      }

      is >> leading;
      if(!is.good() ||
          leading != std::string("<InitialSeeds>")) {
        throw Exception(errors::Configuration)
          << "File \"" << restoreFileName_
          << "\" is ill-structured or otherwise corrupted.\n"
          << "Cannot read beginning of InitialSeeds when restoring random engine states.\n";
      }

      if(seedVectorSize > maxSeeds) {
        throw Exception(errors::Configuration)
          << "File \"" << restoreFileName_
          << "\" is ill-structured or otherwise corrupted.\n"
          << "The number of seeds exceeds 64K.\n";
      }

      readVector(is, seedVectorSize, seedVector);

      is >> trailing;
      if(!is.good() ||
          trailing != std::string("</InitialSeeds>")) {
        throw Exception(errors::Configuration)
          << "File \"" << restoreFileName_
          << "\" is ill-structured or otherwise corrupted.\n"
          << "Cannot read end of InitialSeeds when restoring random engine states.\n";
      }

      is >> leading >> stateVectorSize >> trailing;
      if(!is.good() ||
          leading != std::string("<FullStateLength>") ||
          trailing != std::string("</FullStateLength>")) {
        throw Exception(errors::Configuration)
          << "File \"" << restoreFileName_
          << "\" is ill-structured or otherwise corrupted.\n"
          << "Cannot read state vector length when restoring random engine states.\n";
      }

      is >> leading;
      if(!is.good() ||
          leading != std::string("<FullState>")) {
        throw Exception(errors::Configuration)
          << "File \"" << restoreFileName_
          << "\" is ill-structured or otherwise corrupted.\n"
          << "Cannot read beginning of FullState when restoring random engine states.\n";
      }

      if(stateVectorSize > maxStates) {
        throw Exception(errors::Configuration)
          << "File \"" << restoreFileName_
          << "\" is ill-structured or otherwise corrupted.\n"
          << "The number of states exceeds 64K.\n";
      }

      readVector(is, stateVectorSize, stateVector);

      is >> trailing;
      if(!is.good() ||
          trailing != std::string("</FullState>")) {
        throw Exception(errors::Configuration)
          << "File \"" << restoreFileName_
          << "\" is ill-structured or otherwise corrupted.\n"
          << "Cannot read end of FullState when restoring random engine states.\n";
      }

      if(saveToCache) {
        RandomEngineState randomEngineState;
        randomEngineState.setLabel(moduleLabel);
        std::vector<RandomEngineState>::iterator state =
          std::lower_bound(cache.begin(), cache.end(), randomEngineState);

        if(state != cache.end() && moduleLabel == state->getLabel()) {
          if(seedVector.size() != state->getSeed().size() ||
              stateVector.size() != state->getState().size()) {
            throw Exception(errors::Configuration)
              << "File \"" << restoreFileName_
              << "\" is ill-structured or otherwise corrupted.\n"
              << "Vectors containing engine state are the incorrect size for the type of random engine.\n";
          }
          state->setSeed(seedVector);
          state->setState(stateVector);
        }
      }
      return true;
    }

    void
    RandomNumberGeneratorService::readVector(std::istream& is, unsigned numItems, std::vector<uint32_t>& v) {
      v.clear();
      v.reserve(numItems);
      uint32_t data;
      for(unsigned i = 0; i < numItems; ++i) {
        is >> data;
        if(!is.good()) {
          throw Exception(errors::Configuration)
            << "File \"" << restoreFileName_
            << "\" is ill-structured or otherwise corrupted.\n"
            << "Cannot read vector when restoring random engine states.\n";
        }
        v.push_back(data);
      }
    }

    void
    RandomNumberGeneratorService::startNewSequencesForEvents() {

      if(childIndex_ == 0U && eventSeedOffset_ == 0U) return;

      for(EngineMap::const_iterator iter = engineMap_.begin();
           iter != engineMap_.end();
           ++iter) {

        uint32_t offset1 = childIndex_;
        uint32_t offset2 = eventSeedOffset_;

        std::string const& moduleLabel = iter->first;
        std::string const& engineName = engineNameMap_[moduleLabel];
        VUint32& seeds = seedMap_[moduleLabel];

       if(engineName == std::string("RanecuEngine")) {
          assert(seeds.size() == 2U);
          // Wrap around if the offsets push the seed over the maximum allowed value
          uint32_t mod = maxSeedRanecu + 1U;
          offset1 = offset1 % mod;
          offset2 = offset2 % mod;
          seeds[0] = (seeds[0] + offset1) % mod;
          seeds[0] = (seeds[0] + offset2) % mod;
          long int seedL[2];
          seedL[0] = static_cast<long int>(seeds[0]);
          seedL[1] = static_cast<long int>(seeds[1]);
          iter->second->setSeeds(seedL,0);
        } else {
          assert(seeds.size() == 1U);

          if(engineName == "HepJamesRandom") {
            // Wrap around if the offsets push the seed over the maximum allowed value
            uint32_t mod = maxSeedHepJames + 1U;
            offset1 = offset1 % mod;
            offset2 = offset2 % mod;
            seeds[0] = (seeds[0] + offset1) % mod;
            seeds[0] = (seeds[0] + offset2) % mod;

            long int seedL = static_cast<long int>(seeds[0]);
            iter->second->setSeed(seedL, 0);
          } else {
            assert(engineName == "TRandom3");
            // Wrap around if the offsets push the seed over the maximum allowed value
            // We have to be extra careful with this one because it may also go beyond
            // the values 32 bits can hold
            uint32_t max32 = maxSeedTRandom3;
            if((max32 - seeds[0]) >= offset1) {
              seeds[0] = seeds[0] + offset1;
            } else {
              seeds[0] = offset1 - (max32 - seeds[0]) - 1U;
            }
            if((max32 - seeds[0]) >= offset2) {
              seeds[0] = seeds[0] + offset2;
            } else {
              seeds[0] = offset2 - (max32 - seeds[0]) - 1U;
            }
            long seedL = static_cast<long>(seeds[0]);

            // There is a dangerous conversion from uint32_t to long
            // that occurs above. In the next 2 lines we check the
            // behavior is what we need for the service to work
            // properly.  This conversion is forced on us by the
            // CLHEP and ROOT interfaces. If the assert ever starts
            // to fail we will have to come up with a way to deal
            // with this.
            uint32_t seedu32 = static_cast<uint32_t>(seedL);
            assert(seeds[0] == seedu32);

            iter->second->setSeed(seedL, 0);
          }
        }
      }
    }

    void
    RandomNumberGeneratorService::oldStyleConfig(ParameterSet const& pset) {
      VString pSets = pset.getParameterNamesForType<ParameterSet>();
      for(VString::const_iterator it = pSets.begin(), itEnd = pSets.end(); it != itEnd; ++it) {
        if(*it != std::string("moduleSeeds")) {
          throw Exception(errors::Configuration)
            << "RandomNumberGeneratorService supports two configuration interfaces.\n"
            << "One is old and deprecated, but still supported for backward compatibility\n"
            << "reasons. It is illegal to mix parameters using both the old and new service\n"
            << "interface in the same configuration. It is assumed the old interface is being\n"
            << "used if the parameter set named \"moduleSeeds\" exists.  In that case it is\n"
            << "illegal to have any other nested ParameterSets. This exception was thrown\n"
            << "because that happened.\n";
        }
      }

      ParameterSet const& moduleSeeds = pset.getParameterSet("moduleSeeds");

      std::vector<uint32_t> seeds;

      VString names = moduleSeeds.getParameterNames();
      for(VString::const_iterator itName = names.begin(), itNameEnd = names.end();
           itName != itNameEnd; ++itName) {

        uint32_t seed = moduleSeeds.getUntrackedParameter<uint32_t>(*itName);

        seeds.clear();
        seeds.push_back(seed);
        seedMap_[*itName] = seeds;
        engineNameMap_[*itName] = std::string("HepJamesRandom");

        if(seed > maxSeedHepJames) {
          throw Exception(errors::Configuration)
            << "The CLHEP::HepJamesRandom engine seed should be in the range 0 to 900000000.\n"
            << "The seed passed to the RandomNumberGenerationService from the\n"
               "configuration file was " << seed << ". This was for the module\n"
            << "with label \"" << *itName << "\".";
        }
        long seedL = static_cast<long>(seed);
        engineMap_[*itName] = boost::shared_ptr<CLHEP::HepRandomEngine>(new CLHEP::HepJamesRandom(seedL));
      }
    }
  }
}
