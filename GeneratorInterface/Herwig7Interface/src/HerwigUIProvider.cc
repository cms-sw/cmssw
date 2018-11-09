/**
Marco A. Harrendorf
Dominik Beutel
**/

#include "GeneratorInterface/Herwig7Interface/interface/HerwigUIProvider.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "GeneratorInterface/Core/interface/ParameterCollector.h"
#include "GeneratorInterface/Herwig7Interface/interface/RandomEngineGlue.h"

#include <ThePEG/Utilities/DynamicLoader.h>
#include <ThePEG/Utilities/Debug.h>
#include <ThePEG/Repository/Repository.h>
#include <ThePEG/Handlers/SamplerBase.h>


namespace Herwig {

void HerwigUIProvider::quitWithHelp() const {
edm::LogError("Herwig7Interface") << "An error occured, interface quits now.";
  quit();
}

void HerwigUIProvider::quit() const {
  ThePEG::Repository::cleanup();
  edm::LogInfo("Herwig7Interface") << "Quitting HerwigUIProvider after ThePEG repository cleanup.";
}

HerwigUIProvider::~HerwigUIProvider() {
  ThePEG::Repository::cleanup();
}

HerwigUIProvider::HerwigUIProvider(const edm::ParameterSet &pset, std::string inputFileName, RunMode::Mode runMode) 
  : runMode_(runMode), 
    resume_(false), tics_(true), tag_(),
    inputfile_(inputFileName), repository_(), setupfile_(),
    integrationList_(),
    nEvents_(-1), seed_(0), jobs_(1),
    jobsize_(0), maxjobs_(0)
{

  // check runMode of program and terminate if error state
  if (runMode_ == RunMode::ERROR)
	  edm::LogError("Herwig7Interface") << "Invalid run mode: RunMode::ERROR was passed to Herwig.\n";

/*
  std::string tmpRunMode = pset.getUntrackedParameter<std::string>("runMode", "read");
  if      ( tmpRunMode == "init" )       { runMode_ = RunMode::INIT; }
  else if ( tmpRunMode == "read" )       { runMode_ = RunMode::READ; }
  else if ( tmpRunMode == "build" )      { runMode_ = RunMode::BUILD; }
  else if ( tmpRunMode == "integrate" )  { runMode_ = RunMode::INTEGRATE; }
  else if ( tmpRunMode == "mergegrids" ) { runMode_ = RunMode::MERGEGRIDS; }
  else if ( tmpRunMode == "run" )        { runMode_ = RunMode::RUN; }
  else {
    runMode_ = RunMode::ERROR;
    quitWithHelp();
  }
*/

  // File path to repository file
  repository_  = gen::ParameterCollector::resolve(pset.getParameter<std::string>("repository"));
  if (repository_.empty()) {
    repository_ = std::string("HerwigDefaults.rpo");
  }

  // Number of events
  if ( pset.getUntrackedParameter<int>("numberEvents", -1) != -1 )
    nEvents_ = pset.getUntrackedParameter<int>("numberEvents", 1);


  // run name tag (default given in ggo file)
  if ( pset.getUntrackedParameter<std::string>("runTag", "") != "")
    tag_ = pset.getUntrackedParameter<std::string>("runTag", "Tag1");

  // Debugging level
  if ( pset.getUntrackedParameter<unsigned int>("debugOutput", 0) )
    ThePEG::Debug::setDebug( pset.getUntrackedParameter<unsigned int>("debugOutput", 0) );

  // Floating point exceptions
  if ( pset.getUntrackedParameter<bool>("debugFPE", false) ) 
    ThePEG::Debug::unmaskFpuErrors();

  // Exit-on-error flag
  if ( pset.getUntrackedParameter<bool>("exitOnError", false) )
    ThePEG::Repository::exitOnError() = 1;

  // Tics
  if ( pset.getUntrackedParameter<bool>("hideTics", false) )
    tics_ = false;


 

  // RNG seed
  if ( pset.getUntrackedParameter<int>("seed", 0) != 0 )
    seed_ = pset.getUntrackedParameter<int>("seed", 0);

  // run modification file
  if ( pset.getUntrackedParameter<std::string>("setupFile", "") != "" )
    setupfile_ = pset.getUntrackedParameter<std::string>("setupFile", "");

  // parallel jobs
  if ( pset.getUntrackedParameter<int>("jobs", 1) != 1 )
    jobs_ = pset.getUntrackedParameter<int>("jobs", 1);
  

  // Directories from which Herwig reads filesystem
  std::vector<std::string> aReadDirectories = pset.getUntrackedParameter<std::vector<std::string> >("appendReadDirectories", std::vector<std::string>() );
  std::vector<std::string> pReadDirectories = pset.getUntrackedParameter<std::vector<std::string> >("prependReadDirectories", std::vector<std::string>() );
  appendReadDirectories_.insert(appendReadDirectories_.end(), aReadDirectories.begin(), aReadDirectories.end());
  prependReadDirectories_.insert(prependReadDirectories_.end(), pReadDirectories.begin(), pReadDirectories.end());

  // Library search path for dlopen()
  std::vector<std::string> aPath = pset.getUntrackedParameter<std::vector<std::string> >("appendPath", std::vector<std::string>() );
  std::vector<std::string> pPath = pset.getUntrackedParameter<std::vector<std::string> >("prependPath", std::vector<std::string>() );
  for ( size_t i = 0; i < aPath.size(); ++i )
    ThePEG::DynamicLoader::appendPath( aPath[i] );
  for ( size_t i = 0; i < pPath.size(); ++i )
    ThePEG::DynamicLoader::prependPath( pPath[i] );
 

  // integration list
  if ( pset.getUntrackedParameter<std::string>("integrationList", "") != "" ) {
    integrationList_ = "integrationJob" + pset.getUntrackedParameter<std::string>("integrationList", "1");
  }



  // job size
  if ( pset.getUntrackedParameter<unsigned int>("jobSize", 0) != 0 ) {
    if ( runMode_ == RunMode::BUILD ) {
      jobsize_ = pset.getUntrackedParameter<unsigned int>("jobSize", 1);
      ThePEG::SamplerBase::setIntegratePerJob(jobsize_);
    }
  }

  // max integration jobs
  if ( pset.getUntrackedParameter<unsigned int>("maxJobs", 0) != 0 ) {
    if ( runMode_ == RunMode::BUILD ) {
      maxjobs_ = pset.getUntrackedParameter<unsigned int>("maxJobs", 1);
      ThePEG::SamplerBase::setIntegrationJobs(maxjobs_);
    }
  }


  // Resume
  if ( pset.getUntrackedParameter<bool>("resume", false) )
    resume_ = true;
  

}





void HerwigUIProvider::setRunMode(RunMode::Mode runMode, const edm::ParameterSet &pset, std::string inputFile)
{
	runMode_ = runMode;
	if( !inputFile.empty())
		inputfile_ = inputFile;

	/* If build mode is chosen set these parameters accordingly, else unset them.*/
	if (runMode_ == RunMode::BUILD)
	{
		// job size
		if ( pset.getUntrackedParameter<unsigned int>("jobSize", 0) != 0 )
		{
			jobsize_ = pset.getUntrackedParameter<unsigned int>("jobSize", 1);
			ThePEG::SamplerBase::setIntegratePerJob(jobsize_);
		}
		// max integration jobs
		if ( pset.getUntrackedParameter<unsigned int>("maxJobs", 0) != 0 )
		{
			maxjobs_ = pset.getUntrackedParameter<unsigned int>("maxJobs", 1);
			ThePEG::SamplerBase::setIntegrationJobs(maxjobs_);
		}
	} else {
		jobsize_ = 0;
		ThePEG::SamplerBase::setIntegratePerJob(jobsize_);
		maxjobs_ = 0;
		ThePEG::SamplerBase::setIntegrationJobs(maxjobs_);
	}
}
// End Herwig namespace
}

