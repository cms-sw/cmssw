/** \class ThePEGSource
 *  $Id: ThePEGSource.cc 93 2008-02-25 20:15:36Z stober $
 *  
 *  Oliver Oberst <oberst@ekp.uni-karlsruhe.de>
 *  Fred-Markus Stober <stober@ekp.uni-karlsruhe.de>
 */

#include "GeneratorInterface/ThePEGInterface/interface/ThePEGSource.h"

#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Event.h"

#include "ThePEG/Vectors/HepMCConverter.h"
#include "ThePEG/Utilities/DynamicLoader.h"
#include "ThePEG/Repository/Repository.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/HepMCProduct/interface/GenInfoProduct.h"

#include "HepMC/GenEvent.h"
#include "HepMC/IO_ExtendedAscii.h"

#include <map>
#include <iostream>
#include <sstream>

using namespace std;

namespace ThePEG
{
template<> struct HepMCTraits<HepMC::GenEvent> : public HepMCTraitsBase
	<HepMC::GenEvent, HepMC::GenParticle, HepMC::GenVertex, HepMC::Polarization> {};
}

static string GetFilePath(string filename, string dataLocation)
{
	// TODO: Find better solution
	/* Interpret filename
	 * 1. as absolute path
	 * 2. as relative to external directory
	 * 3. using edm:FileInPath
	 */
	if (filename.find_first_of('/') == 0)
		return filename;
	string external = string(getenv("CMSSW_DATA_PATH")) + "/../external";
	try
	{
		return edm::FileInPath(external + "/" + filename).fullPath();
	} catch (...) {}
	try
	{
		return edm::FileInPath(external + "/" + dataLocation + "/" + filename).fullPath();
	} catch (...) {}
	try
	{
		return edm::FileInPath(filename).fullPath();
	} catch (...) {}
	return edm::FileInPath(dataLocation + "/" + filename).fullPath();
}

void edm::ThePEGSource::InitRepository(const ParameterSet &pset) const
{
	/* Initialize the repository from
	 * 1. the repository file (default values)
	 * 2. the ThePEG config file
	 * 3. the CMSSW config file
	 */
	stringstream logstream;

	// Path to the data directory - place for repo, model and other files
	string dataLocation = pset.getUntrackedParameter<string>("dataLocation", "GeneratorInterface/ThePEGInterface/data");

	// Read the repository of serialized default values
	string repository = pset.getUntrackedParameter<string>("defaultRepository", "HerwigDefaults.rpo");
	if (repository != "")
	{
		repository = GetFilePath(repository, dataLocation);
		edm::LogInfo("ThePEGSource") << "Loading repository (" << repository << ")";
		ThePEG::Repository::load(repository);
	}

	// Read ThePEG config file
	string config = pset.getUntrackedParameter<string>("configFile", "");
	if (config != "")
	{
		if (config.find_first_of('/') != 0)
			config = edm::FileInPath(dataLocation + "/" + config).fullPath();
		edm::LogInfo("ThePEGSource") << "Loading configuration file (" << config << ")";
		ThePEG::Repository::read(config, logstream);
		edm::LogInfo("ThePEGSource") << logstream.str();
	}

	// Read CMSSW config file parameter sets
	ParameterSet thepeg_params = pset.getParameter<ParameterSet>("ThePEGParameters");
	vector<string> param_sets = thepeg_params.getParameter<vector<string> >("parameterSets");

	// Loop over the parameter sets
	for (unsigned i = 0; i < param_sets.size(); ++i)
	{
		string pset = param_sets[i];
		edm::LogInfo("ThePEGSource") << "Loading parameter set (" << pset << ")";

		// Read parameters in the set
		vector<string> params = thepeg_params.getParameter<vector<string> >(pset);

		// Transfer parameters to the repository
		for (vector<string>::const_iterator pIter = params.begin(); pIter != params.end(); ++pIter)
		{
			string out = ThePEG::Repository::exec(*pIter, logstream);
			if (out != "") edm::LogInfo("ThePEGSource") << *pIter << " => " << out;
		}
	}

	// Print the directories where ThePEG looks for libs
	const vector<string> libdirlist = ThePEG::DynamicLoader::allPaths();
	for (vector<string>::const_iterator libdir = libdirlist.begin(); libdir < libdirlist.end(); ++libdir)
		edm::LogInfo("ThePEGSource") << "DynamicLoader path = " << *libdir << endl;

	// Output status information about the repository
	ThePEG::Repository::stats(logstream);
	edm::LogInfo("ThePEGSource") << logstream.str();
}

void edm::ThePEGSource::InitGenerator(const ParameterSet &pset)
{
	// Get generator from the repository and initialize it
	string generatorName = pset.getUntrackedParameter<string>("generatorName", "/Herwig/Generators/LHCGenerator");
	ThePEG::BaseRepository::CheckObjectDirectory(generatorName);
	ThePEG::EGPtr tmp = ThePEG::BaseRepository::GetObject<ThePEG::EGPtr>(generatorName);
	if (tmp)
	{
		string runName = pset.getUntrackedParameter<string>("run", "LHC");
		eg_ = ThePEG::Repository::makeRun(tmp, runName);
		Service<RandomNumberGenerator> rng;
		eg_->setSeed(rng->mySeed());
		eg_->initialize();
		edm::LogInfo("ThePEGSource") << "EventGenerator initialized";
	}
	else
		edm::LogError("ThePEGSource") << "EventGenerator not found!";

	// Skip events
	for (int i = 0; i < pset.getUntrackedParameter<int>("skipEvents", 0); i++)
	{
		eg_->shoot();
		edm::LogInfo("ThePEGSource") << "Event discarded";
	}
}

edm::ThePEGSource::ThePEGSource(const ParameterSet &pset, InputSourceDescription const &desc)
	: GeneratedInputSource(pset, desc)
{  
	InitRepository(pset);
	InitGenerator(pset);

	// Write events in hepmc ascii format for debugging purposes
	string eventLog = pset.getUntrackedParameter<string>("printEvents", "");
	if (eventLog != "")
	{
		iobc_ = new HepMC::IO_ExtendedAscii(eventLog.c_str(), ios::out);
		edm::LogInfo("ThePEGSource") << "Event logging switched on (=> " << eventLog << ")";
	}
	else
		iobc_ = 0;

	produces<HepMCProduct>();
	produces<edm::GenInfoProduct, edm::InRun>();
}

edm::ThePEGSource::~ThePEGSource()
{
	if (eg_) eg_->finalize();
	if (iobc_) delete iobc_;
	edm::LogInfo("ThePEGSource") << "Event generator finalized";
}

bool edm::ThePEGSource::produce(Event &e)
{
	edm::LogInfo("ThePEGSource") << "Start production";

	ThePEG::EventPtr thepeg_event = eg_->shoot();
	if (!thepeg_event)
	{
		edm::LogWarning("ThePEGSource") << "thepeg_event not initialized";
		return false;
	}

	HepMC::GenEvent *hepmc_event = ThePEG::HepMCConverter<HepMC::GenEvent>::convert(*thepeg_event);
	if (!hepmc_event)
	{
		edm::LogWarning("ThePEGSource") << "hepmc_event not initialized";
		return false;
	}

	if (iobc_)
		iobc_->write_event(hepmc_event);

	auto_ptr<HepMCProduct> result(new HepMCProduct());
	result->addHepMCData(hepmc_event);
	e.put(result);
	edm::LogInfo("ThePEGSource") << "Event produced";

	return true;
}

void edm::ThePEGSource::endRun(edm::Run &run)
{
	std::auto_ptr<edm::GenInfoProduct> genInfoProd(new edm::GenInfoProduct);
	genInfoProd->set_cross_section(eg_->integratedXSec());
	run.put(genInfoProd);
}
