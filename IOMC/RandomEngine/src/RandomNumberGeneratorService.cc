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
// $Id$
//

#include "IOMC/RandomEngine/src/RandomNumberGeneratorService.h"

#include <iostream>

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

      try {
        seeds = moduleSeedVectors.getUntrackedParameter<std::vector<uint32_t> >(*itName);
        if (seeds.size() > 0) {
          seedMap_[*itName] = seeds;
        }
      }
      catch (const edm::Exception&) {
        // If there is something wrong, skip it, if this random engine or seed is actually needed
        // an exception will be thrown when the engine or seed is requested and not available
      }
    }
  }
  catch (const edm::Exception&) {
    // It is OK if this is missing.
  }

  try {
    const edm::ParameterSet& moduleSeeds = iPSet.getParameter<edm::ParameterSet>("moduleSeeds");
    
    std::vector<std::string> names = moduleSeeds.getParameterNames();
    for(std::vector<std::string>::const_iterator itName = names.begin(); itName != names.end(); ++itName) {

      // If we already have a seed vector for this label ignore this one
      if (seedMap_.find(*itName) == seedMap_.end()) {

        try {
          uint32_t seed = moduleSeeds.getUntrackedParameter<uint32_t>(*itName);
          seeds.clear();
          seeds.push_back(seed);
          seedMap_[*itName] = seeds;
        }
        catch (const edm::Exception&) {
          // If there is something wrong, skip it, if this random engine or seed is actually needed
          // an exception will be thrown when the engine or seed is requested and not available
        }
      }  
    }
  }
  catch (const edm::Exception&) {
    // It is OK if this is missing.
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

    std::string outputString = "the module with label \"";
    outputString += seedIter->first;
    outputString += "\"";
    if (seedIter->first == sourceLabel) outputString = "the source";

    if (engineName == "HepJamesRandom") {

      if (seedIter->second.size() != 1) {
        throw cms::Exception("RandomNumberFailure")
          << "HepJamesRandom engine requires 1 seed and "
          << seedIter->second.size()
          << " seeds were\n"
          << "specified in the configuration file for "
          << outputString << ".";
      }

      if (seedIter->second[0] > 900000000) {
        throw cms::Exception("RandomNumberFailure")
          << "The HepJamesRandom engine seed should be in the range 0 to 900000000.\n"
          << "The seed passed to the RandomNumberGenerationService from the\n"
             "configuration file was " << seedIter->second[0]
          << " or negative or larger\n"
             "than a 32 bit unsigned integer.  This was for "
          << outputString << ".";
      }
      long seedL = static_cast<long>(seedIter->second[0]);
      CLHEP::HepRandomEngine* engine = new CLHEP::HepJamesRandom(seedL);
      engineMap_[seedIter->first] = engine;
    }
    else {
      throw cms::Exception("RandomNumberFailure")
        << "The configuration file requested the RandomNumberGeneratorService\n"
           "create an unknown random engine type named \""
        << engineName
        << "\"\nfor " << outputString;
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
          << "The module with label \""
          << currentLabel_
          << "\" requested a random number engine from the \n"
             "RandomNumberGeneratorService, but that module was not configured\n"
             "for random numbers.  An engine is created only if a seed(s) is provided\n"
             "in the configuration file.  Please change the configuration\n"
             "file so that the RandomNumberGeneratorService has a random number\n"
             "seed for module \""
          << currentLabel_ << "\" (Make sure the parameter is untracked, has the\n"
             "correct type, and that the parameter name matches the module label).";
      }
      else {
        throw cms::Exception("RandomNumberFailure")
          << "The source requested a random number engine from the \n"
             "RandomNumberGeneratorService, but the source was not configured\n"
             "for random numbers.  An engine is created only if a seed(s) is provided\n"
             "in the configuration file.  Please change the configuration\n"
             "file so that the RandomNumberGeneratorService has a random number\n"
             "seed for the source (Make sure the parameter is untracked and has\n"
             "the correct type).";
      }
    }
    else {
      throw cms::Exception("RandomNumberFailure")
        << "Requested a random number engine from the RandomNumberGeneratorService\n"
           "when no module was active.  This is not supposed to be possible.\n"
           "Please rerun the job using a debugger to get a traceback to show\n"
           "what routines were called and then send information to the edm developers.";
    }
  }  
  return *(currentEngine_->second); 
}

uint32_t
RandomNumberGeneratorService::mySeed() const {

  std::map<std::string, std::vector<uint32_t> >::const_iterator iter;
  iter = seedMap_.find(currentLabel_);

  if (iter == seedMap_.end()) {
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
          << currentLabel_ << "\" (Make sure the parameter is untracked, has the\n"
             "correct type, and that the parameter name matches the module label).";
      }
      else {
        throw cms::Exception("RandomNumberFailure")
          << "The source requested a random number seed from the \n"
             "RandomNumberGeneratorService, but the source was not configured\n"
             "to have a random number seed.  Please change the configuration\n"
             "file so that the RandomNumberGeneratorService has a random number\n"
             "seed for the source (Make sure the parameter is untracked and has\n"
             "the correct type).";
      }
    }
    else {
      throw cms::Exception("RandomNumberFailure")
        << "Requested a random number seed from the RandomNumberGeneratorService\n"
           "when no module was active.  This is not supposed to be possible.\n"
           "Please rerun the job using a debugger to get a traceback to show\n"
           "what routines were called and then send information to the edm developers.";
    }
  }
  return iter->second[0];
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

  try {
    iEvent.getByLabel(restoreStateLabel_, states);
  }
  catch (const edm::Exception&) {
    throw cms::Exception("RandomNumberFailure")
      << "The RandomNumberGeneratorService is trying to restore\n"
      << "the state of the random engines by reading an object from\n"
      << "the event with label \"" << restoreStateLabel_ << "\".  It\n"
      << "fails to find one.  This is probably set to the wrong value\n"
      << "in the configuration file.  It must match the module label\n"
      << "of the RandomEngineStateProducer that created the object in\n"
      << "a previous process\n";   
  }


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

      // Note: It is considered legal for the module label
      // to be in the event but not in the configuration file.
      // In this case we just ignore it and do nothing.

      // The converse is also legal.  If the module label is in the
      // configuration file and not in the event, this is legal.
      // Again we do nothing.

      // Now deal with the case where the module label appears both
      // in the input file and also in the configuration file.

      if (engine != engineMap_.end()) {

        // Check that the engine type in the event and in the configuration
        // file match.
        if (engine->second->name() != std::string("HepJamesRandom")) {

          if (engineLabel == sourceLabel) {
            throw cms::Exception("RandomNumberFailure")
              << "The RandomNumberGeneratorService is trying to restore\n"
              << "the state of the random engines for the source.  An\n"
              << "error was detected because the type of engine in the\n"
	      << "input file and configuration file do not match.\n"
              << "In the configuration file the type is \"" << engine->second->name()
              << "\".\nIn the input file the type is \"HepJamesRandom\".  If\n"
              << "you are not generating any random numbers in the source, then\n"
              << "remove the line in the configuration file that gives the source\n"
              << "a seed and the error will go away.  Otherwise, you must give\n"
              << "the source the same engine type in the configuration file or\n"
              << "stop trying to restore the random engine state.\n";
          }
          else {
            throw cms::Exception("RandomNumberFailure")
              << "The RandomNumberGeneratorService is trying to restore\n"
              << "the state of the random engines for the module \"" 
              << engineLabel << "\".  An\n"
              << "error was detected because the type of engine in the\n"
	      << "input file and configuration file do not match.\n"
              << "In the configuration file the type is \"" << engine->second->name()
              << "\".\nIn the input file the type is \"HepJamesRandom\".  If\n"
              << "you are not generating any random numbers in this module, then\n"
              << "remove the line in the configuration file that gives it\n"
              << "a seed and the error will go away.  Otherwise, you must give\n"
              << "this module the same engine type in the configuration file or\n"
              << "stop trying to restore the random engine state.\n";
          }

        }

        // These two lines actually restore the seed and engine state.
        engine->second->setSeed(engineSeedsL[0], 0);
        engine->second->get(engineStateL);

        seedMap_[engineLabel] = engineSeeds;
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
RandomNumberGeneratorService::print() {
  std::cout << "\n\nRandomNumberGeneratorService dump\n\n";

  std::cout << "    Contents of seedMap\n";
  for (std::map<std::string, std::vector<uint32_t> >::const_iterator iter = seedMap_.begin();
       iter != seedMap_.end();
       ++iter) {
    std::cout << "        " << iter->first;
    std::vector<uint32_t> seeds = iter->second;
    for (std::vector<uint32_t>::const_iterator vIter = seeds.begin();
         vIter != seeds.end();
         ++vIter) {
      std::cout << "   "  << *vIter;
    }
    std::cout << "\n";
  }
  std::cout << "\n    Contents of engineMap\n";
  for (EngineMap::const_iterator iter = engineMap_.begin();
       iter != engineMap_.end();
       ++iter) {
    std::cout << "        " << iter->first
              << "   " << iter->second->name()
              << "   " << iter->second->getSeed()
              << "\n";
  }
  std::cout << "\n";
  std::cout << "    currentLabel_ = " << currentLabel_ << "\n";
  std::cout << "    labelStack_ size = " << labelStack_.size() << "\n";
  int i = 0;
  for (std::vector<std::string>::const_iterator iter = labelStack_.begin();
       iter != labelStack_.end();
       ++iter, ++i) {
    std::cout << "                 " << i << "  " << *iter << "\n";    
  }
  if (currentEngine_ == engineMap_.end()) {
    std::cout << "    currentEngine points to end\n";
  }
  else {
    std::cout << "    currentEngine_ = " << currentEngine_->first
              << "  " << currentEngine_->second->name()
              << "  " << currentEngine_->second->getSeed() << "\n";
  }

  std::cout << "    engineStack_ size = " << engineStack_.size() << "\n";
  i = 0;
  for (std::vector<EngineMap::const_iterator>::const_iterator iter = engineStack_.begin();
       iter != engineStack_.end();
       ++iter, ++i) {
    if (*iter == engineMap_.end()) {
      std::cout << "                 " << i << "  Points to end of engine map\n";    
    }
    else {
      std::cout << "                 " << i << "  " << (*iter)->first 
	        << "  " << (*iter)->second->name() << "  " << (*iter)->second->getSeed() << "\n";    
    }
  }

  std::cout << "    restoreStateLabel_ = " << restoreStateLabel_ << "\n";

  int size = getCachedLabels().size();
  std::cout << "    cachedLabels size = " << size << "\n";
  for (int i = 0; i < size; ++i) {
    std::cout << "      " << i << "  " << getCachedLabels()[i] << "\n";
  }

  size = getCachedStates().size();
  std::cout << "    cachedStates size = " << size << "\n";
  for (int i = 0; i < size; ++i) {
    std::vector<uint32_t> state = getCachedStates()[i];
    std::cout << "      " << i << " : ";
    int j = 0;
    for (std::vector<uint32_t>::const_iterator iter = state.begin();
         iter != state.end();
         ++iter, ++j) {
      if (j > 0 && (j % 10) == 0) {
 	std::cout << "\n              ";
      }
      std::cout << "  " << *iter;
    }
    std::cout << "\n";
  }

  size = getCachedSeeds().size();
  std::cout << "    cachedSeeds size = " << size << "\n";
  for (int i = 0; i < size; ++i) {
    std::vector<uint32_t> seeds = getCachedSeeds()[i];
    std::cout << "      " << i << " : ";
    for (std::vector<uint32_t>::const_iterator iter = seeds.begin();
         iter != seeds.end();
         ++iter) {
      std::cout << "  " << *iter;
    }
    std::cout << "\n";
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
