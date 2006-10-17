// -*- C++ -*-
//
// Package:     Services
// Class  :     RandomNumberGeneratorService
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones, W. David Dagenhart
//         Created:  Tue Mar  7 09:43:46 EST 2006
// $Id: RandomNumberGeneratorService.cc,v 1.2 2006/03/21 20:36:45 chrjones Exp $
//

#include "FWCore/Services/src/RandomNumberGeneratorService.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/RandomEngine/interface/RandomEngineState.h"
#include "CLHEP/Random/JamesRandom.h"
#include "CLHEP/Random/engineIDulong.h"

using namespace edm::service;

static const std::string sourceLabel("@source");

RandomNumberGeneratorService::RandomNumberGeneratorService(const ParameterSet& iPSet,
                                                           ActivityRegistry& iRegistry):
  // The default is an empty string which signals to the process that it should not
  // try to restore the random engines to the state stored in the input event.
  // Otherwise, the configuration file should set this to the module label used
  // in the previous process to store the random engine state
  restoreStateLabel_(iPSet.getUntrackedParameter<std::string>("restoreStateLabel", std::string()))
{
  std::string labels;
  std::vector<uint32_t> seeds;

  // Now get the seeds from the configuration file.  The seeds are used to initialize the
  // random number engines.  Each is associated with either the source or a module label.
  // If there is more than one seed required to initialize the engine type you want to use,
  // the vector form must be used.  Otherwise, either works.  The default engine only requires
  // one seed.  If both the vector seed and single seed commands appear in the configuration
  // file, then the vector form gets used and the other ignored.

  try {
    seeds = iPSet.getUntrackedParameter<std::vector<uint32_t> >("sourceSeedVector");
  }
  // If there is no vector look for a single seed
  catch (const edm::Exception&) {
    try {
      uint32_t seed = iPSet.getUntrackedParameter<uint32_t>("sourceSeed");
      seeds.push_back(seed);
    }
    // OK if you cannot find any
    catch (const edm::Exception&) {
    }
  }
  // If you find seed(s) for the source, save it
  if (seeds.size() > 0) {
    seedMap_[sourceLabel] = seeds;
  }

  try {
    const edm::ParameterSet& moduleSeedVectors = iPSet.getParameter<edm::ParameterSet>("moduleSeedVectors");
    
    std::vector<std::string> names = moduleSeedVectors.getParameterNames();
    for(std::vector<std::string>::const_iterator itName = names.begin(); itName != names.end(); ++itName) {
      seeds = moduleSeedVectors.getUntrackedParameter<std::vector<uint32_t> >(*itName);
      seedMap_[*itName] = seeds;
    }
  }
  catch (const edm::Exception&) {
    // It is OK if this is missing.
  }

  seeds.clear();
  try {
    const edm::ParameterSet& moduleSeeds = iPSet.getParameter<edm::ParameterSet>("moduleSeeds");
    
    std::vector<std::string> names = moduleSeeds.getParameterNames();
    for(std::vector<std::string>::const_iterator itName = names.begin(); itName != names.end(); ++itName) {

      // If we already have a seed vector for this label ignore this one
      if (seedMap_.find(*itName) == seedMap_.end()) {
        uint32_t seed = moduleSeeds.getUntrackedParameter<uint32_t>(*itName);
        seeds.clear();
        seeds.push_back(seed);
        seedMap_[*itName] = seeds;
      }  
    }
  }
  catch (const edm::Exception&) {
    //It is OK if this is missing.
  }

  // Loop over the engines where the seed(s) were specified and see
  // if the engine is also specified.  If not, default to HepJamesRandom.
  // Create the engines and fill the map.
  for (std::map<std::string, std::vector<uint32_t> >::iterator seedIter = seedMap_.begin();
       seedIter != seedMap_.end();
       ++seedIter) {

    // Initialize with default engine
    std::string engineName = "HepJamesRandom";
    try {
      const edm::ParameterSet& moduleEngines = iPSet.getParameter<edm::ParameterSet>("moduleEngines");
      engineName = moduleEngines.getUntrackedParameter<std::string>(seedIter->first);
    }
    catch (const edm::Exception&) {
      // OK if none, use default
    }
    if (engineName == "HepJamesRandom") {
      if (seedIter->second.size() != 1) {
        throw cms::Exception("RandomNumberFailure")
          << "HepJamesRandom engine requires one seed and "
          << seedIter->second.size()
          << " were specified in the configuration file for module label \""
          << seedIter->first << "\"";
      }
      if (seedIter->second[0] > 900000000) {
        throw cms::Exception("RandomNumberFailure")
          << "HepJamesRandom engine seed should be in the range 0 to 900000000.  "
          << "Seed was " << seedIter->second[0]
          << " in the configuration file for module label \""
          << seedIter->first << "\"";
      }
      long seedL = static_cast<long>(seedIter->second[0]);
      CLHEP::HepRandomEngine* engine = new CLHEP::HepJamesRandom(seedL);
      engineMap_[seedIter->first] = engine;
    }
    else {
      throw cms::Exception("RandomNumberFailure")
        << "Configuration file requested unknown random engine type: "
        << engineName;
    }
  }

  iRegistry.watchPostBeginJob(this,&RandomNumberGeneratorService::postBeginJob);
  iRegistry.watchPostEndJob(this,&RandomNumberGeneratorService::postEndJob);

  iRegistry.watchPreModuleConstruction(this,&RandomNumberGeneratorService::preModuleConstruction);
  iRegistry.watchPostModuleConstruction(this,&RandomNumberGeneratorService::postModuleConstruction);

  iRegistry.watchPreSourceConstruction(this,&RandomNumberGeneratorService::preSourceConstruction);
  iRegistry.watchPostSourceConstruction(this,&RandomNumberGeneratorService::postSourceConstruction);

  iRegistry.watchPreProcessEvent(this,&RandomNumberGeneratorService::preEventProcessing);
  iRegistry.watchPostProcessEvent(this,&RandomNumberGeneratorService::postEventProcessing);
   
  iRegistry.watchPreModule(this,&RandomNumberGeneratorService::preModule);
  iRegistry.watchPostModule(this,&RandomNumberGeneratorService::postModule);

  // the default for the stack is to point to the 'end' of our map which is used to define not set
  engineStack_.push_back(engineMap_.end());
  currentEngine_ = engineMap_.end();

  labelStack_.push_back(std::string());
  currentLabel_ = std::string();
}

RandomNumberGeneratorService::~RandomNumberGeneratorService()
{
  // Delete the engines
  for (EngineMap::iterator iter = engineMap_.begin();
       iter != engineMap_.end();
       ++iter) {
    delete iter->second;
  }
}

CLHEP::HepRandomEngine& 
RandomNumberGeneratorService::getEngine() const {

  if (currentEngine_ == engineMap_.end()) {
    if (currentLabel_ != std::string() ) {
      if ( currentLabel_ != sourceLabel) {
        throw cms::Exception("RandomNumberFailure")
          << "requested a random number engine for a module with label \""
          << currentLabel_
          << "\" that was not configured to use a random number.  "
             "Please change configuration file so that the "
             "RandomNumberGeneratorService has a random number seed for \""
          << currentLabel_ << "\"";
      }
      else {
        throw cms::Exception("RandomNumberFailure")
          << "requested a random number engine for the source but the "
             "source was not configured to use a random number.  "
             "Please change configuration file so that the "
             "RandomNumberGeneratorService has a random "
             "number seed for the source";
      }
    }
    else {
      throw cms::Exception("RandomNumberFailure")
        << "Requested a random number engine when no module was active.  "
           "This is not supposed to be possible.  "
           "Please rerun the job using a debugger to get a traceback "
           "to show what routines were called and then send information "
           "to the edm developers.";
    }
  }  
  return *(currentEngine_->second); 
}

uint32_t
RandomNumberGeneratorService::mySeed() const {

  if (currentEngine_ == engineMap_.end()) {
    if (currentLabel_ != std::string() ) {
      if ( currentLabel_ != sourceLabel) {
        throw cms::Exception("RandomNumberFailure")
          << "The module with label \""
          << currentLabel_
          << "\" requested a random number seed from the \n"
             "RandomNumberGeneratorService, but that module was not configured\n"
             "to have a random number seed.  Please change the configuration\n"
             "file so that the RandomNumberGeneratorService has a random number\n"
             "seed for module \""
          << currentLabel_ << "\".";
      }
      else {
        throw cms::Exception("RandomNumberFailure")
          << "requested a random number seed for the source but the "
             "source was not configured to use a random number.  "
             "Please change configuration file so that the "
             "RandomNumberGeneratorService has a random "
             "number seed for the source";
      }
    }
    else {
      throw cms::Exception("RandomNumberFailure")
        << "Requested a random number seed when no module was active.  "
           "This is not supposed to be possible.  "
           "Please rerun the job using a debugger to get a traceback "
           "to show what routines were called and then send information "
           "to the edm developers.";
    }
  }
  std::map<std::string, std::vector<uint32_t> >::const_iterator iter;
  iter = seedMap_.find(currentLabel_);
  std::vector<uint32_t> seeds = iter->second;
  return seeds[0];
}

void 
RandomNumberGeneratorService::preModuleConstruction(const ModuleDescription& iDesc)
{
  push(iDesc.moduleLabel_);
}

void 
RandomNumberGeneratorService::postModuleConstruction(const ModuleDescription&)
{
  pop();
}

void 
RandomNumberGeneratorService::preSourceConstruction(const ModuleDescription& iDesc)
{
  push(sourceLabel);
}

void 
RandomNumberGeneratorService::postSourceConstruction(const ModuleDescription&)
{
  pop();
}

void 
RandomNumberGeneratorService::postBeginJob()
{
  //finished begin run so waiting for first event and the source will be the first one called
  push(sourceLabel);
}

void 
RandomNumberGeneratorService::postEndJob()
{
  if (labelStack_.size() != 1) {
    pop();
  }
}

void 
RandomNumberGeneratorService::preEventProcessing(const edm::EventID&, const edm::Timestamp&)
{
  //finished with source and now waiting for a module
  pop();
}

void 
RandomNumberGeneratorService::postEventProcessing(const Event&, const EventSetup&)
{
  //finished processing the event so should start another one soon.  The first thing to be called will be the source
  push(sourceLabel);
}

void 
RandomNumberGeneratorService::preModule(const ModuleDescription& iDesc)
{
  push(iDesc.moduleLabel_);
}

void 
RandomNumberGeneratorService::postModule(const ModuleDescription&)
{
  pop();
}

const std::vector<std::string>& 
RandomNumberGeneratorService::getCachedLabels() const {

  return cachedLabels_;
}

const std::vector<std::vector<uint32_t> >&
RandomNumberGeneratorService::getCachedStates() const {

  return cachedStates_;
}

const std::vector<std::vector<uint32_t> >&
RandomNumberGeneratorService::getCachedSeeds() const {

  return cachedSeeds_;
}

void
RandomNumberGeneratorService::restoreState(const edm::Event& iEvent) {

  if ( restoreStateLabel_ == std::string()) return;

  Handle<std::vector<RandomEngineState> > states;
  iEvent.getByLabel(restoreStateLabel_, states);

  // Get the information out of the persistent object into
  // convenient vectors and convert to the type CLHEP requires.
  // There may be some issues here when we go to 64 bit machines.

  for (std::vector<RandomEngineState>::const_iterator iter = states->begin();
       iter != states->end();
       ++iter) {
    std::string engineLabel = iter->getLabel();
    std::vector<uint32_t> engineState = iter->getState();
    std::vector<unsigned long> engineStateL;
    for (std::vector<uint32_t>::const_iterator iVal = engineState.begin();
         iVal != engineState.end();
         ++iVal) {
      engineStateL.push_back(static_cast<unsigned long>(*iVal));
    }

    std::vector<uint32_t> engineSeeds = iter->getSeed();
    std::vector<long> engineSeedsL;
    for (std::vector<uint32_t>::const_iterator iVal = engineSeeds.begin();
         iVal != engineSeeds.end();
         ++iVal) {
      engineSeedsL.push_back(static_cast<long>(*iVal));
    }

    // We need to handle each type of engine differently because each
    // has different requirements on the seed or seeds.
    if (engineStateL[0] == CLHEP::engineIDulong<HepJamesRandom>()) {
      EngineMap::iterator engine = engineMap_.find(engineLabel);

      // If this module label is in the event but was not initialized in the
      // configuration file ignore it.  This is good because it should be legal
      // to only rerun some of modules previously run.  An exception will be
      // thrown when the module tries to get the engine if it needs to be there.
      if (engine != engineMap_.end()) {

        // Check that the engine type in the event and in the configuration
        // file match.
        if (engine->second->name() != std::string("HepJamesRandom")) {
          throw cms::Exception("RandomNumberFailure")
            << "For module \"" << engineLabel
            << "\".  configuration random engine type \"" << engine->second->name()
            << "\".  Attempting to restore random state from input file, but previous "
            << "engine was of type \"HepJamesRandom\".  This is an error because the "
            << "engine types must be the same to do the restore.  To make this work "
            << "modify the configuration file so they either match or you are not attempting "
            << " restore the random state.";
        }

        // These two lines actually restore the seed and engine state.
        engine->second->setSeed(engineSeedsL[0], 0);
        engine->second->get(engineStateL);
      }
    }
    // This should not be possible because this code should be able to restore
    // any kind of engine whose state can be saved.
    else {
      throw cms::Exception("RandomNumberFailure")
        << "Requested unknown random engine when restoring random state";
    }
  }
}

void
RandomNumberGeneratorService::snapShot()
{
  cachedLabels_.clear();
  cachedStates_.clear();
  cachedSeeds_.clear();

  // Loop over the engines and copy the engine state,
  // labels, and initial seeds into temporary cache
  // for later use by the RandomEngineStateProducer module
  for (EngineMap::const_iterator iter = engineMap_.begin();
       iter != engineMap_.end();
       ++iter) {

    cachedLabels_.push_back(iter->first);

    std::vector<unsigned long> stateL = iter->second->put();
    std::vector<uint32_t> state32;
    for (std::vector<unsigned long>::const_iterator vIter = stateL.begin();
         vIter != stateL.end();
         ++vIter) {
      state32.push_back(static_cast<uint32_t>(*vIter));
    }
    cachedStates_.push_back(state32);

    cachedSeeds_.push_back(seedMap_[iter->first]);
  }
}

void
RandomNumberGeneratorService::push(const std::string& iLabel)
{
  currentEngine_ = engineMap_.find(iLabel);
  engineStack_.push_back(currentEngine_);
  
  labelStack_.push_back(iLabel);
  currentLabel_ = iLabel;
}

void
RandomNumberGeneratorService::pop()
{
  engineStack_.pop_back();
  //NOTE: algorithm is such that we always have at least one item in the stacks
  currentEngine_ = engineStack_.back();
  labelStack_.pop_back();
  currentLabel_ = labelStack_.back();
}
