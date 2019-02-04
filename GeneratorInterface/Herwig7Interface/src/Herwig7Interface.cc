/** \class Herwig7Interface
 *  
 *  Marco A. Harrendorf marco.harrendorf@cern.ch
 *  Dominik Beutel dominik.beutel@cern.ch
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <memory>
#include <cmath>
#include <cstdlib>

#include <algorithm>

#include <boost/shared_ptr.hpp>
#include <boost/filesystem.hpp>

#include <HepMC/GenEvent.h>
#include <HepMC/PdfInfo.h>
#include <HepMC/IO_GenEvent.h>

#include <Herwig/API/HerwigAPI.h>

#include <ThePEG/Utilities/DynamicLoader.h>
#include <ThePEG/Repository/Repository.h>
#include <ThePEG/Handlers/EventHandler.h>
#include <ThePEG/Handlers/XComb.h>
#include <ThePEG/EventRecord/Event.h>
#include <ThePEG/EventRecord/Particle.h> 
#include <ThePEG/EventRecord/Collision.h>
#include <ThePEG/EventRecord/TmpTransform.h>
#include <ThePEG/Config/ThePEG.h>
#include <ThePEG/PDF/PartonExtractor.h>
#include <ThePEG/PDF/PDFBase.h>
#include <ThePEG/Utilities/UtilityBase.h>
#include <ThePEG/Vectors/HepMCConverter.h>


#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "GeneratorInterface/Core/interface/ParameterCollector.h"

#include "GeneratorInterface/Herwig7Interface/interface/Proxy.h"
#include "GeneratorInterface/Herwig7Interface/interface/RandomEngineGlue.h"
#include "GeneratorInterface/Herwig7Interface/interface/Herwig7Interface.h"

#include "CLHEP/Random/RandomEngine.h"

using namespace std;
using namespace gen;

Herwig7Interface::Herwig7Interface(const edm::ParameterSet &pset) :
	randomEngineGlueProxy_(ThePEG::RandomEngineGlue::Proxy::create()),
	dataLocation_(ParameterCollector::resolve(pset.getParameter<string>("dataLocation"))),
	generator_(pset.getParameter<string>("generatorModule")),
	run_(pset.getParameter<string>("run")),
	dumpConfig_(pset.getUntrackedParameter<string>("dumpConfig", "HerwigConfig.in")),
	skipEvents_(pset.getUntrackedParameter<unsigned int>("skipEvents", 0))
{
	// Write events in hepmc ascii format for debugging purposes
	string dumpEvents = pset.getUntrackedParameter<string>("dumpEvents", "");
	if (!dumpEvents.empty()) {
		iobc_.reset(new HepMC::IO_GenEvent(dumpEvents, ios::out));
		edm::LogInfo("ThePEGSource") << "Event logging switched on (=> " << dumpEvents << ")";
	}
	// Clear dumpConfig target
	if (!dumpConfig_.empty())
		ofstream cfgDump(dumpConfig_.c_str(), ios_base::trunc);
}

Herwig7Interface::~Herwig7Interface () noexcept
{
	if (eg_)
		eg_->finalize();
	edm::LogInfo("Herwig7Interface") << "Event generator finalized";
}

void Herwig7Interface::setPEGRandomEngine(CLHEP::HepRandomEngine* v) {
    
        randomEngineGlueProxy_->setRandomEngine(v);
        randomEngine = v;
        ThePEG::RandomEngineGlue *rnd = randomEngineGlueProxy_->getInstance();
        if(rnd) {
            rnd->setRandomEngine(v);
        }
}


void Herwig7Interface::initRepository(const edm::ParameterSet &pset)
{
	std::string runModeTemp = pset.getUntrackedParameter<string>("runModeList","read,run");

	// To Lower
	std::transform(runModeTemp.begin(), runModeTemp.end(), runModeTemp.begin(), ::tolower);



	while(!runModeTemp.empty())
	{
		// Split first part of List
		std::string choice;
		size_t pos = runModeTemp.find(",");
        if (std::string::npos == pos)
            choice=runModeTemp;
        else
		    choice = runModeTemp.substr(0, pos);
        
		if (pos == std::string::npos)
			runModeTemp.erase();
        else
            runModeTemp.erase(0, pos+1);

		// construct HerwigUIProvider object and return it as global object
		HwUI_ = new Herwig::HerwigUIProvider(pset, dumpConfig_, Herwig::RunMode::READ);
		edm::LogInfo("Herwig7Interface") << "HerwigUIProvider object with run mode " << HwUI_->runMode() << " created.\n";


		// Chose run mode
		if	( choice == "read" )
		{
			createInputFile(pset);
			HwUI_->setRunMode(Herwig::RunMode::READ, pset, dumpConfig_);
			edm::LogInfo("Herwig7Interface") << "Input file " << dumpConfig_ << " will be passed to Herwig for the read step.\n";
			callHerwigGenerator();
		}
		else if	( choice == "build" )
		{
			createInputFile(pset);
			HwUI_->setRunMode(Herwig::RunMode::BUILD, pset, dumpConfig_);
			edm::LogInfo("Herwig7Interface") << "Input file " << dumpConfig_ << " will be passed to Herwig for the build step.\n";
			callHerwigGenerator();

		}
		else if	( choice == "integrate" )
		{
			std::string runFileName = run_ + ".run";
			edm::LogInfo("Herwig7Interface") << "Run file " << runFileName << " will be passed to Herwig for the integrate step.\n";
			HwUI_->setRunMode(Herwig::RunMode::INTEGRATE, pset, runFileName);
			callHerwigGenerator();

		}
		else if	( choice == "run" )
		{
			std::string runFileName = run_ + ".run";
			edm::LogInfo("Herwig7Interface") << "Run file " << runFileName << " will be passed to Herwig for the run step.\n";
			HwUI_->setRunMode(Herwig::RunMode::RUN, pset, runFileName);
		}
		else
		{
			edm::LogInfo("Herwig7Interface") << "Cannot recognize \"" << choice << "\".\n"
							 << "Trying to skip step.\n";
			continue;
		}

	}

}

void Herwig7Interface::callHerwigGenerator()
{
  try {

    edm::LogInfo("Herwig7Interface") << "callHerwigGenerator function invoked with run mode " << HwUI_->runMode() << ".\n";

    // Call program switches according to runMode
    switch ( HwUI_->runMode() ) {
    case Herwig::RunMode::INIT:        Herwig::API::init(*HwUI_);       break;
    case Herwig::RunMode::READ:        Herwig::API::read(*HwUI_);       break;
    case Herwig::RunMode::BUILD:       Herwig::API::build(*HwUI_);      break;
    case Herwig::RunMode::INTEGRATE:   Herwig::API::integrate(*HwUI_);  break;
    case Herwig::RunMode::MERGEGRIDS:  Herwig::API::mergegrids(*HwUI_); break;
    case Herwig::RunMode::RUN:         {    
                                            HwUI_->setSeed(randomEngine->getSeed());
                                            eg_ =  Herwig::API::prepareRun(*HwUI_); break;}
    case Herwig::RunMode::ERROR:       
      edm::LogError("Herwig7Interface") << "Error during read in of command line parameters.\n"
                << "Program execution will stop now."; 
      return;
    default:          		     
      HwUI_->quitWithHelp();
    }

    return;

  }
  catch ( ThePEG::Exception & e ) {
    edm::LogError("Herwig7Interface") << ": ThePEG::Exception caught.\n"
              << e.what() << '\n'
      	      << "See logfile for details.\n";
    return;
  }
  catch ( std::exception & e ) {
    edm::LogError("Herwig7Interface") << ": " << e.what() << '\n';
    return;
  }
  catch ( const char* what ) {
    edm::LogError("Herwig7Interface") << ": caught exception: "
	      << what << "\n";
    return;
  }

}


bool Herwig7Interface::initGenerator()
{
	if ( HwUI_->runMode() == Herwig::RunMode::RUN) {
		edm::LogInfo("Herwig7Interface") << "Starting EventGenerator initialization";
		callHerwigGenerator();
		edm::LogInfo("Herwig7Interface") << "EventGenerator initialized";

		// Skip events
		for (unsigned int i = 0; i < skipEvents_; i++) {
			flushRandomNumberGenerator();
			eg_->shoot();
			edm::LogInfo("Herwig7Interface") << "Event discarded";
		}

		return true;

	} else {
		edm::LogInfo("Herwig7Interface") << "Stopped EventGenerator due to missing run mode.";
		return false;
/*
		throw cms::Exception("Herwig7Interface")
			<< "EventGenerator could not be initialized due to wrong run mode!" << endl;
*/
	}

}

void Herwig7Interface::flushRandomNumberGenerator()
{
	/*ThePEG::RandomEngineGlue *rnd = randomEngineGlueProxy_->getInstance();

	if (!rnd)
		edm::LogWarning("ProxyMissing")
			<< "ThePEG not initialised with RandomEngineGlue.";
	else
		rnd->flush();
      */
}

unique_ptr<HepMC::GenEvent> Herwig7Interface::convert(
					const ThePEG::EventPtr &event)
{
	return std::unique_ptr<HepMC::GenEvent>(
		ThePEG::HepMCConverter<HepMC::GenEvent>::convert(*event));
}




double Herwig7Interface::pthat(const ThePEG::EventPtr &event)
{
	using namespace ThePEG;

	if (!event->primaryCollision())
		return -1.0;

	tSubProPtr sub = event->primaryCollision()->primarySubProcess();
	TmpTransform<tSubProPtr> tmp(sub, Utilities::getBoostToCM(
							sub->incoming()));

	double pthat = (*sub->outgoing().begin())->momentum().perp() / ThePEG::GeV;
	for(PVector::const_iterator it = sub->outgoing().begin();
	    it != sub->outgoing().end(); ++it)
		pthat = std::min<double>(pthat, (*it)->momentum().perp() / ThePEG::GeV);

	return pthat;
}




void Herwig7Interface::createInputFile(const edm::ParameterSet &pset)
{
	/* Initialize the input config for Herwig from
	 * 1. the Herwig7 config files
	 * 2. the CMSSW config blocks
	 * Writes them to an output file which is read by Herwig
	*/

	stringstream logstream;


	// Contains input config passed to Herwig
	stringstream herwiginputconfig;

	// Define output file to which input config is written, too, if dumpConfig parameter is set. 
	// Otherwise use default file HerwigConfig.in which is read in by Herwig
	ofstream cfgDump;
	cfgDump.open(dumpConfig_.c_str(), ios_base::app);
	


	// Read Herwig config files as input
	vector<string> configFiles = pset.getParameter<vector<string> >("configFiles");
	// Loop over the config files
	for ( const auto & iter : configFiles ) {
		// Open external config file
		ifstream externalConfigFile (iter);
		if (externalConfigFile.is_open()) {
			edm::LogInfo("Herwig7Interface") << "Reading config file (" << iter << ")" << endl;
			stringstream configFileStream;
			configFileStream << externalConfigFile.rdbuf();
			string configFileContent = configFileStream.str();
			
			// Comment out occurence of saverun in config file since it is set later considering run and generator option
			string searchKeyword("saverun");
   			if(configFileContent.find(searchKeyword) !=std::string::npos) {
				edm::LogInfo("Herwig7Interface") << "Commented out saverun command in external input config file(" << iter << ")" << endl;
				configFileContent.insert(configFileContent.find(searchKeyword),"#");
			}
			herwiginputconfig << "# Begin Config file input" << endl  << configFileContent << endl << "# End Config file input";
			edm::LogInfo("Herwig7Interface") << "Finished reading config file (" << iter << ")" << endl;
		}
		else {
			edm::LogWarning("Herwig7Interface") << "Could not read config file (" << iter << ")" << endl;
		}
	}

	edm::LogInfo("Herwig7Interface") << "Start with processing CMSSW config" << endl;
	// Read CMSSW config file parameter sets starting from "parameterSets"
	ParameterCollector collector(pset);
	ParameterCollector::const_iterator iter;
	iter = collector.begin();
	herwiginputconfig << endl << "# Begin Parameter set input\n" << endl;
	for(; iter != collector.end(); ++iter) {
		herwiginputconfig << *iter << endl;
	}

	// Add some additional necessary lines to the Herwig input config
	herwiginputconfig << "saverun " << run_ << " " << generator_ << endl;
	// write the ProxyID for the RandomEngineGlue to fill its pointer in
	ostringstream ss;
	ss << randomEngineGlueProxy_->getID();
	//herwiginputconfig << "set " << generator_ << ":RandomNumberGenerator:ProxyID " << ss.str() << endl;


	// Dump Herwig input config to file, so that it can be read by Herwig
	cfgDump << herwiginputconfig.str() << endl;
	cfgDump.close();
}

