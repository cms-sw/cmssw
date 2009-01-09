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

#include <iostream>
#include <fstream>
#include <limits>
#include <unistd.h>

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/RandomEngine/interface/RandomEngineState.h"
#include "CLHEP/Random/JamesRandom.h"
#include "CLHEP/Random/RanecuEngine.h"
#include "CLHEP/Random/engineIDulong.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "IOMC/RandomEngine/src/TRandomAdaptor.h"

using namespace edm::service;

typedef std::vector<std::string> VString;
typedef std::vector<uint32_t>    Vuint32;
typedef edm::ParameterSet        PSet;

static const std::string sourceLabel("@source");


RandomNumberGeneratorService::RandomNumberGeneratorService(const ParameterSet& iPSet,
                                                           ActivityRegistry& iRegistry):
  // The default is an empty string which signals to the process that it should not
  // try to restore the random engines to the state stored in the input event.
  // Otherwise, the configuration file should set this to the module label used
  // in the previous process to store the random engine state
  restoreStateLabel_(iPSet.getUntrackedParameter<std::string>("restoreStateLabel", std::string())),
  saveFileName_(std::string()),
  saveFileNameRecorded_(false),
  restoreFileName_(std::string()),
  oldStyle_(false)
{

  std::string labels;
  std::vector<uint32_t> seeds;
  PSet  empty_PSet;
  VString  empty_VString;
  Vuint32  empty_Vuint32;

  // Record the names of the files to use to store engine states at the end of
  // each event and to restore engine states at the beginning of a job.
  // A blank name means don't bother.
  saveFileName_    = iPSet.getUntrackedParameter<std::string>("saveFileName",   std::string());
  restoreFileName_ = iPSet.getUntrackedParameter<std::string>("restoreFileName",std::string());

// The saveFileName and must correspond to a file name without any path specification.
// Throw if that is not true.
  if(!saveFileName_.empty() && (saveFileName_.find("/") != std::string::npos)) {
    std::ostringstream sstr;
    sstr << "Configuration: The saveFileName parameter " << saveFileName_ << "\n" 
         << "must be a simple file name with no path specification.\n" ;
    edm::Exception except(edm::errors::Configuration, sstr.str());
    throw except;
  }     

// Check if the configuration file is still expressed in the old style.
// We do this by looking for a PSet named moduleSeeds.  That parameter is 
// unique to an old style cfg file.

  VString pSets = iPSet.getParameterNamesForType<edm::ParameterSet>();
  for(VString::const_iterator it = pSets.begin(), itEnd = pSets.end(); it != itEnd; ++it) {
    if(*it == std::string("moduleSeeds")) oldStyle_ = true;
  }

// If only the source needs a random engine, there won't be a PSet named moduleSeeds and
// this test will fail.  To cover that case, look for a sourceSeed parameter as well.  

  if(iPSet.getUntrackedParameter<uint32_t>("sourceSeed",0) != 0) oldStyle_ = true;
      
  if(oldStyle_) {
    oldStyleConfig(iPSet);
  } else {

// We have a new style configuration file. deal with it here.
// Loop over parameters of type ParameterSet and look for things
// that should not appear in new-style cfg files. If any are found,
// it means the user has supplied a mixture of new- and old-style
// configuration parameters that can't possibly work correctly.

    bool source;
    std::string engineName;
    uint32_t initialSeed;
    std::vector<uint32_t> initialSeedSet;

    std::string sourceEngine = iPSet.getUntrackedParameter<std::string>("sourceEngine",std::string(""));
    if(!sourceEngine.empty()) {
      throw edm::Exception(edm::errors::Configuration)
        << "sourceEngine is obsolete for new-style configuration files." 
        << "\nPlease switch to the newer cfg format for the RandomNumberGeneratorService." ;
    }

    uint32_t sourceSeed = iPSet.getUntrackedParameter<uint32_t>("sourceSeed",0);
    if(sourceSeed != 0) {
      throw edm::Exception(edm::errors::Configuration)
        << "sourceSeed is obsolete for new-style configuration files." 
        << "\nPlease switch to the newer cfg format for the RandomNumberGeneratorService." ;
    }

    std::vector<uint32_t> sourceSeedVector = iPSet.getUntrackedParameter<std::vector<uint32_t> >("sourceSeedVector",empty_Vuint32);
    if(!sourceSeedVector.empty()) {
      throw edm::Exception(edm::errors::Configuration)
        << "sourceSeedVector is obsolete for new-style configuration files." 
        << "\nPlease switch to the newer cfg format for the RandomNumberGeneratorService." ;
    }

    VString pSets = iPSet.getParameterNamesForType<edm::ParameterSet>();
    for(VString::const_iterator it = pSets.begin(), itEnd = pSets.end(); it != itEnd; ++it) {
      source = false;
      if(*it == std::string("moduleEngines")) {
        throw edm::Exception(edm::errors::Configuration)
          << "moduleEngines is obsolete for new-style configuration files." 
          << "\nPlease switch to the newer cfg format for the RandomNumberGeneratorService." ;
      }
  
      if(*it == std::string("moduleSeedVectors")) {
        throw edm::Exception(edm::errors::Configuration)
          << "moduleSeedVectors is obsolete for new-style configuration files." 
          << "\nPlease switch to the newer cfg format for the RandomNumberGeneratorService." ;
      }
  
  // Deal with a source engine here
      if(*it == std::string("theSource")) source = true;
  
      PSet secondary = iPSet.getParameter<edm::ParameterSet>(*it);
      engineName = secondary.getUntrackedParameter<std::string>("engineName",std::string("HepJamesRandom"));
      if(!isEngineNameValid(engineName)) {
        throw edm::Exception(edm::errors::Configuration)
          << "The specified Random Engine name, " << engineName
          << " does not correspond to a supported engine." ;
      }
  
  // For the RanecuEngine case, require a seed set containing exactly two seeds.
  
      if(engineName == std::string("RanecuEngine")) {
        if(secondary.exists("initialSeedSet")) {
          initialSeedSet = secondary.getUntrackedParameter<std::vector<uint32_t> >("initialSeedSet",empty_Vuint32);
          uint32_t nSeeds = initialSeedSet.size();
          uint32_t mSeeds = expectedSeedCount(engineName);
          if(nSeeds != mSeeds) {
            throw edm::Exception(edm::errors::Configuration)
              << "RanecuEngine requires 2 seeds and "
              << nSeeds << " seeds were\n"
              << "specified in the configuration file for "
              << "the module with label " << *it << "." ;
          }
          if(source) {
            seedMap_[sourceLabel] = initialSeedSet;
          } else {
            seedMap_[*it] = initialSeedSet;
          }
          CLHEP::HepRandomEngine* engine = new RanecuEngine();
          if(source) {
            engineMap_[sourceLabel] = engine;
          } else {
            engineMap_[*it] = engine;
          }
          if (initialSeedSet[0] > std::numeric_limits<uint32_t>::max() ||
              initialSeedSet[1] > std::numeric_limits<uint32_t>::max()) {  // They need to fit in a 32 bit integer
            throw edm::Exception(edm::errors::Configuration)
              << "The RanecuEngine seeds should be in the range 0 to " << std::numeric_limits<uint32_t>::max() << " .\n"
              << "The seeds passed to the RandomNumberGenerationService from the\n"
                 "configuration file were " << initialSeedSet[0] << " and " << initialSeedSet[1]
              << " (or one was negative\nor larger "
                 "than a 32 bit unsigned integer).\nThis was for "
              << "the module with label " << *it << ".";
          }
          long int seedL[2];
          seedL[0] = static_cast<long int>(initialSeedSet[0]);
          seedL[1] = static_cast<long int>(initialSeedSet[1]);
          engine->setSeeds(seedL,0);
        } else {
          throw edm::Exception(edm::errors::Configuration)
            << "No initialSeedSet was supplied for engine " << engineName
            << ". Aborting." ;
        }
  
  // For the other engines, require one seed each as follows:
  //   If a single seed is offered, accept it unconditionally.
  //   If a seed set is offered use the first element of the set unconditionally.
  //   If both are offered, preferentially use the first element of the set.
  
      } else {
        initialSeed = 0;
        if(secondary.exists("initialSeed"))  {
          initialSeed = secondary.getUntrackedParameter<uint32_t>("initialSeed",0);
        }
        if(secondary.exists("initialSeedSet"))  {
          initialSeedSet = secondary.getUntrackedParameter<std::vector<uint32_t> >("initialSeedSet",empty_Vuint32);
          initialSeed = initialSeedSet[0];
        }
        if(initialSeed == 0)  { 
          throw edm::Exception(edm::errors::Configuration)
            << "No initialSeed was supplied for engine " << engineName
            << ". Aborting." ;
        }
        seeds.clear();
        seeds.push_back(initialSeed);
        if(source) {
          seedMap_[sourceLabel] = seeds;
        } else {
          seedMap_[*it] = seeds;
        }
        if(engineName == "HepJamesRandom") {
          if (initialSeed > 900000000) {
            throw edm::Exception(edm::errors::Configuration)
              << "The HepJamesRandom engine seed should be in the range 0 to 900000000.\n"
              << "The seed passed to the RandomNumberGenerationService from the\n"
                 "configuration file was " << initialSeed
              << " or negative or larger\n"
                 "than a 32 bit unsigned integer.  This was for "
              << "the module with label " << *it << ".";
          }
          long int seedL = static_cast<long int>(initialSeed);
          CLHEP::HepRandomEngine* engine = new HepJamesRandom(seedL);
          if(source) {
            engineMap_[sourceLabel] = engine;
          } else {
            engineMap_[*it] = engine;
          }
        } else if(engineName == "TRandom3") {
          CLHEP::HepRandomEngine* engine = new TRandomAdaptor(initialSeed);
          if(source) {
            engineMap_[sourceLabel] = engine;
          } else {
            engineMap_[*it] = engine;
          }
        }
      }
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

  iRegistry.watchPreModuleBeginJob(this,&RandomNumberGeneratorService::preModuleBeginJob);
  iRegistry.watchPostModuleBeginJob(this,&RandomNumberGeneratorService::postModuleBeginJob);

  iRegistry.watchPreModuleEndJob(this,&RandomNumberGeneratorService::preModuleEndJob);
  iRegistry.watchPostModuleEndJob(this,&RandomNumberGeneratorService::postModuleEndJob);

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
       if(oldStyle_) {
          throw edm::Exception(edm::errors::Configuration)
            << "The module with label \""
            << currentLabel_
            << "\" requested a random number engine from the \n"
               "RandomNumberGeneratorService, but that module was not configured\n"
               "for random numbers.  An engine is created only if a seed(s) is provided\n"
               "in the configuration file.  Please add the following line to the\n"
               "configuration file in the moduleSeeds PSet of the RandomNumberGeneratorService:\n"
               "        untracked uint32 " << currentLabel_ << " = <your random number seed>\n\n"
               "Adding the line above will not work if you have modified the default and\n"
               "selected a different type of random engine that requires two or more seeds.\n"
               "In that case, you will need to add the following line to the moduleSeedVectors\n"
               "PSet instead:\n"
	       "        untracked vuint32 " << currentLabel_ 
               << " = { <your first random number seed>, <your second random number seed> ... }\n\n";
        } else {
          throw edm::Exception(edm::errors::Configuration)
            << "The module with label \""
            << currentLabel_
            << "\" requested a random number engine from the \n"
               "RandomNumberGeneratorService, but that module was not configured\n"
               "for random numbers.  An engine is created only if a seed(s) is provided\n"
               "in the configuration file.  Please add the following PSet to the\n"
               "configuration file for the RandomNumberGeneratorService:\n"
               "        PSet " << currentLabel_ << " = {\n"
               "                   untracked string engineName = <name of chosen engine>\n"
               "                   untracked uint32 initialSeed = <value of chosen seed>\n" 
               "                  }\n"
               "Adding the line above will not work if you have modified the default and\n"
               "selected a different type of random engine that requires two or more seeds.\n"
               "In that case, you will need to replace the line with initialSeed with\n"
	       "                   untracked vuint32 initialSeedSet = { <your first random number seed>,"
               "<your second random number seed> ... }\n\n";
        }
      } else {
        if(oldStyle_) {
          throw edm::Exception(edm::errors::Configuration)
            << "The source requested a random number engine from the \n"
               "RandomNumberGeneratorService, but the source was not configured\n"
               "for random numbers.  An engine is created only if a seed(s) is provided\n"
               "in the configuration file.  Please add the following line to the\n"
               "configuration file in the RandomNumberGeneratorService block:\n"
               "        untracked uint32 sourceSeed = <your random number seed>\n\n"
               "Adding the line above will not work if you have modified the default and\n"
               "selected a different type of random engine that requires two or more seeds.\n"
               "In that case, you will need to add the following line instead:\n"
               "        untracked vuint32 sourceSeedVector = { <your first random number seed>, <your second random number seed> ... }\n\n";
        } else {
          throw edm::Exception(edm::errors::Configuration)
            << "The source requested a random number engine from the \n"
               "RandomNumberGeneratorService, but the source was not configured\n"
               "for random numbers.  An engine is created only if a seed(s) is provided\n"
               "in the configuration file.  Please add the following PSet to the\n"
               "configuration file in the RandomNumberGeneratorService block:\n"
               "        PSet theSource  = {\n"
               "                   untracked string engineName = <name of chosen engine>\n"
               "                   untracked uint32 initialSeed = <value of chosen seed>\n" 
               "                  }\n"
               "Adding the line above will not work if you have modified the default and\n"
               "selected a different type of random engine that requires two or more seeds.\n"
               "In that case, you will need to add the following line instead:\n"
               "        untracked vuint32 sourceSeedVector = { <your first random number seed>, <your second random number seed> ... }\n\n";
        }
      }
    }
    else {
      throw edm::Exception(edm::errors::Unknown)
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
       if(oldStyle_) {
          throw edm::Exception(edm::errors::Configuration)
            << "The module with label \""
            << currentLabel_
            << "\" requested a random number engine from the \n"
               "RandomNumberGeneratorService, but that module was not configured\n"
               "for random numbers.  An engine is created only if a seed(s) is provided\n"
               "in the configuration file.  Please add the following line to the\n"
               "configuration file in the moduleSeeds PSet of the RandomNumberGeneratorService:\n"
               "        untracked uint32 " << currentLabel_ << " = <your random number seed>\n\n"
               "Adding the line above will not work if you have modified the default and\n"
               "selected a different type of random engine that requires two or more seeds.\n"
               "In that case, you will need to add the following line to the moduleSeedVectors\n"
               "PSet instead:\n"
	       "        untracked vuint32 " << currentLabel_ 
               << " = { <your first random number seed>, <your second random number seed> ... }\n\n";
        } else {
          throw edm::Exception(edm::errors::Configuration)
            << "The module with label \""
            << currentLabel_
            << "\" requested a random number engine from the \n"
               "RandomNumberGeneratorService, but that module was not configured\n"
               "for random numbers.  An engine is created only if a seed(s) is provided\n"
               "in the configuration file.  Please add the following PSet to the\n"
               "configuration file for the RandomNumberGeneratorService:\n"
               "        PSet " << currentLabel_ << " = {\n"
               "                   untracked string engineName = <name of chosen engine>\n"
               "                   untracked uint32 initialSeed = <value of chosen seed>\n" 
               "                  }\n"
               "Adding the line above will not work if you have modified the default and\n"
               "selected a different type of random engine that requires two or more seeds.\n"
               "In that case, you will need to replace the line with initialSeed with\n"
	       "                   untracked vuint32 initialSeedSet = { <your first random number seed>,"
               "<your second random number seed> ... }\n\n";
        }
      } else {
        if(oldStyle_) {
          throw edm::Exception(edm::errors::Configuration)
            << "The source requested a random number engine from the \n"
               "RandomNumberGeneratorService, but the source was not configured\n"
               "for random numbers.  An engine is created only if a seed(s) is provided\n"
               "in the configuration file.  Please add the following line to the\n"
               "configuration file in the RandomNumberGeneratorService block:\n"
               "        untracked uint32 sourceSeed = <your random number seed>\n\n"
               "Adding the line above will not work if you have modified the default and\n"
               "selected a different type of random engine that requires two or more seeds.\n"
               "In that case, you will need to add the following line instead:\n"
               "        untracked vuint32 sourceSeedVector = { <your first random number seed>, <your second random number seed> ... }\n\n";
        } else {
          throw edm::Exception(edm::errors::Configuration)
            << "The source requested a random number engine from the \n"
               "RandomNumberGeneratorService, but the source was not configured\n"
               "for random numbers.  An engine is created only if a seed(s) is provided\n"
               "in the configuration file.  Please add the following PSet to the\n"
               "configuration file in the RandomNumberGeneratorService block:\n"
               "        PSet theSource  = {\n"
               "                   untracked string engineName = <name of chosen engine>\n"
               "                   untracked uint32 initialSeed = <value of chosen seed>\n" 
               "                  }\n"
               "Adding the line above will not work if you have modified the default and\n"
               "selected a different type of random engine that requires two or more seeds.\n"
               "In that case, you will need to add the following line instead:\n"
               "        untracked vuint32 sourceSeedVector = { <your first random number seed>, <your second random number seed> ... }\n\n";
        }
      }
    } else {
      throw edm::Exception(edm::errors::Unknown)
        << "Requested a random number engine from the RandomNumberGeneratorService\n"
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
  push(iDesc.moduleLabel());
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

  // If there is an engine state file, us it to restore all engines to that state.
  if(!restoreFileName_.empty()) restoreEngineState(restoreFileName_);
 
  // Here is the right place to record engine states if that has been requested
  if(!saveFileName_.empty())  {
    saveEngineState(saveFileName_);
    if(!saveFileNameRecorded_) {
      std::string fullName = constructSaveFileName();
      Service<JobReport> reportSvc;
      reportSvc->reportRandomStateFile(fullName);
      saveFileNameRecorded_ = true;
    }
  }
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
  // Here is the right place to record engine states if that has been requested
  if(!saveFileName_.empty())  {
    saveEngineState(saveFileName_);
    if(!saveFileNameRecorded_) {
      std::string fullName = constructSaveFileName();
      Service<JobReport> reportSvc;
      reportSvc->reportRandomStateFile(fullName);
      saveFileNameRecorded_ = true;
    }
  }
  //finished processing the event so should start another one soon.  The first thing to be called will be the source
  push(sourceLabel);
}

void 
RandomNumberGeneratorService::preModule(const ModuleDescription& iDesc)
{
  push(iDesc.moduleLabel());
}

void 
RandomNumberGeneratorService::postModule(const ModuleDescription&)
{
  pop();
}

void 
RandomNumberGeneratorService::preModuleBeginJob(const ModuleDescription& iDesc)
{
  push(iDesc.moduleLabel());
}

void 
RandomNumberGeneratorService::postModuleBeginJob(const ModuleDescription&)
{
  pop();
}

void 
RandomNumberGeneratorService::preModuleEndJob(const ModuleDescription& iDesc)
{
  push(iDesc.moduleLabel());
}

void 
RandomNumberGeneratorService::postModuleEndJob(const ModuleDescription&)
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
    if(!states.isValid()) {
    throw edm::Exception(edm::errors::ProductNotFound)
      << "The RandomNumberGeneratorService is trying to restore\n"
      << "the state of the random engines by reading an object from\n"
      << "the event with label \"" << restoreStateLabel_ << "\".  It\n"
      << "fails to find one.  This is probably set to the wrong value\n"
      << "in the configuration file.  It must match the module label\n"
      << "of the RandomEngineStateProducer that created the object in\n"
      << "a previous process\n";   
  }

  // The service does not currently support restoring the state
  // of the source random engine.  Throw an exception if the user tries.
  // If we ever try to implement this, the main step is getting the functions
  // restoreState and snapShot called at the correct point(s) in time in the 
  // source.  And you also need a source that both reads an input file and
  // generates random numbers.  If those things are done, I think one could
  // simply delete the following error check and everything should
  // work.  The state of the source is already saved, the problem
  // involves restoring it.  Currently, the calls to snapShot and restoreState
  // are in the base class InputSource and also snapShot gets called at the
  // end of this function.  Alternatively, one could imagine separating the
  // parts of the source that generate random numbers and putting those
  // parts in normal modules.
  EngineMap::iterator sourceEngine = engineMap_.find(sourceLabel);
  if (sourceEngine != engineMap_.end()) {
    throw edm::Exception(edm::errors::UnimplementedFeature)
      << "The RandomNumberGeneratorService was instructed to restore\n"
      << "the state of the random engines and the source is\n"
      << "configured to generate random numbers.\n"
      << "Currently the RandomNumberGeneratorService is not\n"
      << "capable of restoring the random engine state of the\n"
      << "source.  This feature has not been implemented yet.\n"
      << "You need to remove the line in the configuration file\n"
      << "that gives the source a seed or stop trying to restore\n"
      << "the state of the random numbers (remove the line that sets\n"
      << "the \"restoreStateLabel\" in the configuration file).\n";
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

      seedMap_[engineLabel] = engineSeeds;

      // We need to handle each type of engine differently because each
      // has different requirements on the seed or seeds.
      if (engineStateL[0] == CLHEP::engineIDulong<CLHEP::HepJamesRandom>()) {

        checkEngineType(engine->second->name(), std::string("HepJamesRandom"), engineLabel);

        // These two lines actually restore the seed and engine state.
        engine->second->setSeed(engineSeedsL[0], 0);
        engine->second->get(engineStateL);
      }
      else if (engineStateL[0] == CLHEP::engineIDulong<CLHEP::RanecuEngine>()) {

        checkEngineType(engine->second->name(), std::string("RanecuEngine"), engineLabel);

        // This line actually restores the engine state.
        engine->second->get(engineStateL);
      }

      else if (engineStateL[0] == CLHEP::engineIDulong<TRandomAdaptor>()) {

        checkEngineType(engine->second->name(), std::string("TRandom3"), engineLabel);

        // This line actually restores the engine state.
        engine->second->setSeed(engineSeedsL[0], 0);
        engine->second->get(engineStateL);
      }
      // This should not be possible because this code should be able to restore
      // any kind of engine whose state can be saved.
      else {
        throw edm::Exception(edm::errors::Unknown)
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
  snapShot();
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

void
RandomNumberGeneratorService::checkEngineType(const std::string& typeFromConfig,
                                              const std::string& typeFromEvent,
                                              const std::string& engineLabel)
{
  if (typeFromConfig != typeFromEvent) {

    if (engineLabel == sourceLabel) {
      throw edm::Exception(edm::errors::Configuration)
        << "The RandomNumberGeneratorService is trying to restore\n"
        << "the state of the random engine for the source.  An\n"
        << "error was detected because the type of the engine in the\n"
	<< "input file and the configuration file do not match.\n"
        << "In the configuration file the type is \"" << typeFromConfig
        << "\".\nIn the input file the type is \"" << typeFromEvent << "\".  If\n"
        << "you are not generating any random numbers in the source, then\n"
        << "remove the line in the configuration file that gives the source\n"
        << "a seed and the error will go away.  Otherwise, you must give\n"
        << "the source the same engine type in the configuration file or\n"
        << "stop trying to restore the random engine state.\n";
    }
    else {
      throw edm::Exception(edm::errors::Configuration)
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
}

void
RandomNumberGeneratorService::dumpVector(const std::vector<uint32_t> &v)
{
  if(v.empty()) return;
  size_t numItems = v.size();
  for( int i=0; i<(int)numItems; ++i)  {
    if( i != 0 && i%10 == 0 )  std::cerr << std::endl;
    std::cerr << std::setw(13) << v[i] ;
  }
    if(numItems%10 != 0)  std::cerr << std::endl;
}

void
RandomNumberGeneratorService::stashVector(const std::vector<unsigned long> &v,
                                                std::ostream &outFile)
{
  if(v.empty()) return;
  size_t numItems = v.size();
  for( int i=0; i<(int)numItems; ++i)  {
    if( i != 0 && i%10 == 0 )  outFile << "\n" ;
    outFile << std::setw(13) << v[i] ;
  }
    if(numItems%10 != 0)  outFile << "\n" ;
}

std::vector<unsigned long>
RandomNumberGeneratorService::restoreVector(std::istream &is, const int32_t numItems)
{
  unsigned long value;
  std::vector<unsigned long> v;
  std::ostringstream sstr;
  int i = 0;
  is >> std::ws;
  while(i<numItems)  {
    if(!(is >> value)) {
      sstr << "Configuration: Unable to read the random engine state vector from file, "
           << restoreFileName_ << ".\n" 
           << "The file has probably been corrupted.\n" ;
      edm::Exception except(edm::errors::Configuration, sstr.str());
      throw except;
    } else {
      v.push_back(value);
      ++i;
    }
  }
  return v;
}


void RandomNumberGeneratorService::restoreEngineState(const std::string& fileName)
{

// Check that we do or do not want to restore state from a previously written state file

  std::ifstream inFile;
  std::ostringstream sstr;
  inFile.open(fileName.c_str(), std::ifstream::in);
  if(!inFile) {
    sstr << "Configuration: Unable to open the file, "
         << fileName << ", to restore the engine status.\n" ;
    edm::Exception except(edm::errors::Configuration, sstr.str());
    throw except;
  }

  std::string text;
  inFile >> text;
  if(text != std::string("<RandomEngineStates>")) {
    sstr << "Configuration: File, " << restoreFileName_ 
         << " is ill-structured or otherwise corrupted.\n"
         << "Cannot read the file header word.\n" ;
    edm::Exception except(edm::errors::Configuration, sstr.str());
    throw except;
  }
  while(processStanza(inFile)) { }
}

bool RandomNumberGeneratorService::processStanza(std::istream &is)
{
  std::string leading, trailing;
  std::string moduleLabel;
  std::string engineName;
  long initialSeed;
  int stateVectorLength;
  std::vector<unsigned long> stateVector;
  std::ostringstream sstr;

  is >> leading;
  if(leading == std::string("</RandomEngineStates>")) return false;
  is >> moduleLabel >> trailing;
  if(leading != std::string("<ModuleLabel>") || trailing != std::string("</ModuleLabel>")) {
    sstr << "Configuration: File, " << restoreFileName_ 
         << ", is ill-structured or otherwise corrupted.\n" 
         << "Cannot read a module label.\n" ;
    edm::Exception except(edm::errors::Configuration, sstr.str());
    throw except;
  }

  is >> leading >> engineName >> trailing;
  if(leading != std::string("<EngineName>")  || trailing != std::string("</EngineName>")) {
    sstr << "Configuration: File, " << restoreFileName_ 
         << ", is ill-structured or otherwise corrupted.\n" 
         << "Cannot read an engine name.\n" ;
    edm::Exception except(edm::errors::Configuration, sstr.str());
    throw except;
  }

  is >> leading >> initialSeed >> trailing;
  if(leading != std::string("<InitialSeed>") || trailing != std::string("</InitialSeed>")) {
    sstr << "Configuration: File, " << restoreFileName_ 
         << ", is ill-structured or otherwise corrupted.\n" 
         << "Cannot read initial seed(s).\n" ;
    edm::Exception except(edm::errors::Configuration, sstr.str());
    throw except;
  }

  is >> leading >> stateVectorLength >> trailing;
  if(leading != std::string("<FullStateLength>") || trailing != std::string("</FullStateLength>")) {
    sstr << "Configuration: File, " << restoreFileName_ 
         << ", is ill-structured or otherwise corrupted.\n" 
         << "Cannot read the state vector length.\n" ;
    edm::Exception except(edm::errors::Configuration, sstr.str());
    throw except;
  }  

  is >> leading;
  if(leading != std::string("<FullState>")) {
    sstr << "Configuration: File, " << restoreFileName_ 
         << ", is ill-structured or otherwise corrupted.\n"
         << "Cannot read state vector header.\n" ;
    edm::Exception except(edm::errors::Configuration, sstr.str());
    throw except;
  }  
  stateVector = restoreVector(is, stateVectorLength);
  is >> trailing;
  if(trailing != std::string("</FullState>")) {
    sstr << "Configuration: File, " << restoreFileName_ 
         << ", is ill-structured or otherwise corrupted.\n"
         << "Cannot read state vector trailer.\n" ;
    edm::Exception except(edm::errors::Configuration, sstr.str());
    throw except;
  }  
 
// We've recovered everything.  Now make sure it's all consistent
// and, if so, overwrite the state.

  CLHEP::HepRandomEngine* engine;
  EngineMap::iterator iter = engineMap_.find(moduleLabel);
  if(iter == engineMap_.end()) {
    if(moduleLabel == sourceLabel) {

//    Damned special case!
//    For now, we have suppressed writing this stanza of the state save file
//    so we should never get here.  If we do get here, yell and throw.

      sstr << "Configuration: sourceLabel @source is currently disallowed"
           << "\nbut appears in save file, " << saveFileName_ << "\n" ;
      edm::Exception except(edm::errors::Configuration, sstr.str());
      throw except;
    } else {
      sstr << "Configuration: The module label, " << moduleLabel 
           << ", found in the state save file does not match any label "
           << "in the configuration file.  There is a mismatch between "
           << "the configuration file and the state save file, "
           << saveFileName_ << "\n" ;
      edm::Exception except(edm::errors::Configuration, sstr.str());
      throw except;
    } 
  } else {
    engine = iter->second;
    if(engine->name() != engineName) {
      sstr << "Configuration: The engine name in the configuration file (" << engine->name() 
           << ")\ndoes not match the name (" << engineName << ") in the state save file,\n" 
           << saveFileName_ << ", for the module labelled " << moduleLabel << "\n" ;
      edm::Exception except(edm::errors::Configuration, sstr.str());
      throw except;
    }
    engine->setSeed(initialSeed,0);
    bool success = engine->get(stateVector);
    if(!success) {
      sstr << "Configuration: Setting the state of " << engine->name() << " failed.\n"
           << "Aborting.\n" ;
      edm::Exception except(edm::errors::Configuration, sstr.str());
      throw except;
    }
  }
  return true;
}

void RandomNumberGeneratorService::saveEngineState(const std::string& fileName)
{
  std::ofstream outFile;
  outFile.open(fileName.c_str(), std::ofstream::out | std::ofstream::trunc);
  if(!outFile) {
    std::ostringstream sstr;
    sstr << "Configuration: Unable to open the file, "
         << fileName << ", to save the engine status.\n" ;
    edm::Exception except(edm::errors::Configuration, sstr.str());
    throw except;
  }
  outFile << "<RandomEngineStates>\n" ;
  for(EngineMap::const_iterator iter = engineMap_.begin(), iterEnd = engineMap_.end();
    iter != iterEnd;
    ++iter) {
    std::string moduleLabel = iter->first;
//
//  The service does not support restoring the state of the source engine. Until that
//  is dealt with, there is no point in even saving the source engine state so, for
//  the present, we skip it.
//
    if(moduleLabel != sourceLabel) {
      std::string engineName = iter->second->name();
      long initialSeed = iter->second->getSeed();
      std::vector<unsigned long> stateVector = iter->second->put();
      int stateVectorLength = stateVector.size();

      outFile << "<ModuleLabel>\n" << moduleLabel << "\n</ModuleLabel>\n" ;
      outFile << "<EngineName>\n" << engineName << "\n</EngineName>\n" ;
      outFile << "<InitialSeed>\n" << initialSeed << "\n</InitialSeed>\n" ;
      outFile << "<FullStateLength>\n" << stateVectorLength << "\n</FullStateLength>\n" ;
      outFile << "<FullState>\n" ;
      stashVector(stateVector, outFile);
      outFile   << "</FullState>\n" ;
    }
  }
  outFile << "</RandomEngineStates>\n" ;
}

bool RandomNumberGeneratorService::isEngineNameValid(std::string const &name)
{
  bool valid = false;
  if(name == std::string("HepJamesRandom")) valid = true;
  if(name == std::string("RanecuEngine"))   valid = true;
  if(name == std::string("TRandom3"))       valid = true;
  return valid;
}

int32_t RandomNumberGeneratorService::expectedSeedCount(std::string const &name)
{
  int32_t count = 0;
  if(name == std::string("HepJamesRandom")) count = 1;
  if(name == std::string("RanecuEngine"))   count = 2;
  if(name == std::string("TRandom3"))       count = 1;
  return count;
}
std::string RandomNumberGeneratorService::constructSaveFileName()
{
  std::string fullName;
  fullName.reserve(1000);
  fullName = getcwd(&fullName[0],1000); 
  fullName += "/" + saveFileName_;
  fullName.reserve();
  return fullName;
}

void
RandomNumberGeneratorService::oldStyleConfig(const ParameterSet& iPSet)
{
  std::string labels;
  std::vector<uint32_t> seeds;
  PSet  empty_PSet;
  VString  empty_VString;
  Vuint32  empty_Vuint32;

  std::string engineName;

  VString pSets = iPSet.getParameterNamesForType<edm::ParameterSet>();
  for(VString::const_iterator it = pSets.begin(), itEnd = pSets.end(); it != itEnd; ++it) {
    if(*it == std::string("theSource")) {
      throw edm::Exception(edm::errors::Configuration)
        << "PSet theSource is not recognized for old-style configuration files." 
        << "\nYou must be consistent in your use of new- or old-style cfg format"
        << "\nfor the RandomNumberGeneratorService." ;
    }
  }

  // If there is more than one seed required to initialize the engine type you want to use,
  // the vector form must be used.  Otherwise, either works.  The default engine only requires
  // one seed.  If both the vector seed and single seed commands appear in the configuration
  // file, then the vector form gets used and the other ignored.

  if(iPSet.exists("sourceSeedVector")) {
    seeds = iPSet.getUntrackedParameter<std::vector<uint32_t> >("sourceSeedVector");
  // If there is no vector look for a single seed
  } else if(iPSet.exists("sourceSeed")) {
      uint32_t seed = iPSet.getUntrackedParameter<uint32_t>("sourceSeed");
      seeds.push_back(seed);
  }

  // If you find seed(s) for the source, save it
  if (seeds.size() > 0) {
    seedMap_[sourceLabel] = seeds;
  }

  if(iPSet.exists("moduleSeedVectors")) {
    const edm::ParameterSet& moduleSeedVectors = iPSet.getParameter<edm::ParameterSet>("moduleSeedVectors");
    
    std::vector<std::string> names = moduleSeedVectors.getParameterNames();
    for(std::vector<std::string>::const_iterator itName = names.begin(), itNameEnd = names.end(); itName != itNameEnd; ++itName) {

      if(seedMap_.find(*itName) == seedMap_.end()) {
        if(moduleSeedVectors.exists(*itName)) {
          seeds = moduleSeedVectors.getUntrackedParameter<std::vector<uint32_t> >(*itName);
          if (seeds.size() > 0) {
            seedMap_[*itName] = seeds;
          }
        }
      }
    }
        // If there is something wrong, skip it. If this random engine or seed is actually needed
        // an exception will be thrown when the engine or seed is requested and not available
  }
    // It is OK if this is missing.

  if(iPSet.exists("moduleSeeds")) {
    const edm::ParameterSet& moduleSeeds = iPSet.getParameter<edm::ParameterSet>("moduleSeeds");
    
    std::vector<std::string> names = moduleSeeds.getParameterNames();
    for(std::vector<std::string>::const_iterator itName = names.begin(), itNameEnd = names.end();
                                                 itName != itNameEnd; ++itName) {

    // If we already have a seed vector for this label ignore this one
      if (seedMap_.find(*itName) == seedMap_.end()) {

        if(moduleSeeds.exists(*itName)) {
          uint32_t seed = moduleSeeds.getUntrackedParameter<uint32_t>(*itName);
          seeds.clear();
          seeds.push_back(seed);
          seedMap_[*itName] = seeds;
        }
        // If there is something wrong, skip it, if this random engine or seed is actually needed
        // an exception will be thrown when the engine or seed is requested and not available
      }
    }  
  }

    // It is OK if this is missing.

  // Loop over the engines where the seed(s) were specified and see
  // if the engine is also specified.  If not, default to HepJamesRandom.
  // Create the engines and fill the map.

  for (std::map<std::string, std::vector<uint32_t> >::iterator seedIter  = seedMap_.begin();
                                                               seedIter != seedMap_.end();
                                                             ++seedIter) {

    // Initialize with default engine
    std::string engineName = "HepJamesRandom";

    // Go see if this module name already appears in the engineMap_ as a consequence of
    // the new cfg format. Enter this engine only if it isn't already there.
    if(engineMap_.find(seedIter->first) == engineMap_.end()) {
 
      if (seedIter->first == sourceLabel) {
        if(iPSet.exists("sourceEngine")) {
          engineName = iPSet.getUntrackedParameter<std::string>("sourceEngine");
        }
      } else if(iPSet.exists("moduleEngines")) {
        const edm::ParameterSet& moduleEngines = iPSet.getParameter<edm::ParameterSet>("moduleEngines");
        if(moduleEngines.exists(seedIter->first)) {
          engineName = moduleEngines.getUntrackedParameter<std::string>(seedIter->first);
        }
      }
      // OK if none, use default

      std::string outputString = "the module with label \"";
      outputString += seedIter->first;
      outputString += "\"";
      if (seedIter->first == sourceLabel) outputString = "the source";

      if (engineName == "HepJamesRandom") {

        if (seedIter->second.size() != 1) {
          throw edm::Exception(edm::errors::Configuration)
            << "HepJamesRandom engine requires 1 seed and "
            << seedIter->second.size()
            << " seeds were\n"
            << "specified in the configuration file for "
            << outputString << ".";
        }

        if (seedIter->second[0] > 900000000) {
          throw edm::Exception(edm::errors::Configuration)
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
      } else if (engineName == "RanecuEngine") {

        if (seedIter->second.size() != 2) {
          throw edm::Exception(edm::errors::Configuration)
            << "RanecuEngine requires 2 seeds and "
            << seedIter->second.size()
            << " seeds were\n"
            << "specified in the configuration file for "
            << outputString << ".";
        }

        if (seedIter->second[0] > std::numeric_limits<uint32_t>::max() ||
            seedIter->second[1] > std::numeric_limits<uint32_t>::max()) {  // They need to fit in a 31 bit integer
          throw edm::Exception(edm::errors::Configuration)
            << "The RanecuEngine seeds should be in the range 0 to " << std::numeric_limits<uint32_t>::max() << ".\n"
            << "The seeds passed to the RandomNumberGenerationService from the\n"
   	       "configuration file were " << seedIter->second[0] << " and " << seedIter->second[1]
            << " (or one was negative\nor larger "
               "than a 32 bit unsigned integer).\nThis was for "
            << outputString << ".";
        }
        long seedL[2];
        seedL[0] = static_cast<long>(seedIter->second[0]);
        seedL[1] = static_cast<long>(seedIter->second[1]);
        CLHEP::HepRandomEngine* engine = new CLHEP::RanecuEngine();
        engine->setSeeds(seedL, 0);
        engineMap_[seedIter->first] = engine;

      } else if (engineName == "TRandom3") {

        if (seedIter->second.size() != 1) {
          throw edm::Exception(edm::errors::Configuration)
            << "TRandom3 engine requires 1 seed and "
            << seedIter->second.size()
            << " seeds were\n"
            << "specified in the configuration file for "
            << outputString << ".";
        }
        long seedL = static_cast<long>(seedIter->second[0]);
        CLHEP::HepRandomEngine* engine = new TRandomAdaptor(seedL);
        engineMap_[seedIter->first] = engine;
      } else {
        throw edm::Exception(edm::errors::Configuration)
          << "The configuration file requested the RandomNumberGeneratorService\n"
             "create an unknown random engine type named \""
          << engineName
          << "\"\nfor " << outputString
          << "\nCurrently the only valid types are HepJamesRandom, RanecuEngine and TRandom3";
      }
    }
  }
}
