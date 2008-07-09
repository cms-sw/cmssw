/** \class ThePEGInterface
 *  $Id: ThePEGInterface.cc,v 1.4 2008/07/09 10:00:22 stober Exp $
 *  
 *  Oliver Oberst <oberst@ekp.uni-karlsruhe.de>
 *  Fred-Markus Stober <stober@ekp.uni-karlsruhe.de>
 */

#include <iostream>
#include <sstream>
#include <cstdlib>
#include <memory>
#include <cmath>
#include <stdlib.h>

#include <boost/filesystem.hpp>

#include <HepMC/GenEvent.h>
#include <HepMC/PdfInfo.h>
#include <HepMC/IO_ExtendedAscii.h>

#include <ThePEG/Utilities/DynamicLoader.h>
#include <ThePEG/Repository/Repository.h>
#include <ThePEG/Handlers/EventHandler.h>
#include <ThePEG/Handlers/XComb.h>
#include <ThePEG/EventRecord/Event.h>
#include <ThePEG/EventRecord/Particle.h> 
#include <ThePEG/EventRecord/Collision.h>
#include <ThePEG/Config/ThePEG.h>
#include <ThePEG/PDF/PartonExtractor.h>
#include <ThePEG/PDF/PDFBase.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "GeneratorInterface/ThePEGInterface/interface/ThePEGInterface.h"
#include "GeneratorInterface/ThePEGInterface/interface/HepMCConverter.h"

using namespace std;

ThePEGInterface::ThePEGInterface(const edm::ParameterSet &pset) :
	dataLocation_(resolveEnvVars(pset.getParameter<string>("dataLocation"))),
	generator_(pset.getParameter<string>("generatorModule")),
	run_(pset.getParameter<string>("run")),
	configDump_(pset.getUntrackedParameter<string>("configDump", "")),
	skipEvents_(pset.getUntrackedParameter<int>("skipEvents", 0))
{
	// Write events in hepmc ascii format for debugging purposes
	string eventLog = pset.getUntrackedParameter<string>("printEvents", "");
	if (!eventLog.empty()) {
		iobc_.reset(new HepMC::IO_ExtendedAscii(eventLog.c_str(), ios::out));
		edm::LogInfo("ThePEGSource") << "Event logging switched on (=> " << eventLog << ")";
	}
	// Clear configDump target
	if (!configDump_.empty())
		ofstream cfgDump(configDump_.c_str(), ios_base::trunc);
}

ThePEGInterface::~ThePEGInterface()
{
	if (eg_)
		eg_->finalize();
	edm::LogInfo("ThePEGInterface") << "Event generator finalized";
}

string ThePEGInterface::dataFile(const string &fileName) const
{
	return dataLocation_ + "/" + fileName;
}

string ThePEGInterface::dataFile(const edm::ParameterSet &pset,
	                         const string &paramName) const
{
	const edm::Entry &entry = pset.retrieve(paramName);
	if (entry.typeCode() == 'F')
		return entry.getFileInPath().fullPath();
	else
		return dataFile(entry.getString());
}

string ThePEGInterface::resolveEnvVars(const string &s)
{
	string result(s);

	for(;;) {
		string::size_type pos = result.find("${");
		if (pos == string::npos)
			break;

		string::size_type endpos = result.find('}', pos);
		if (endpos == string::npos)
			break;
		else
			++endpos;

		string var = result.substr(pos + 2, endpos - pos - 3);
		const char *path = getenv(var.c_str());

		result.replace(pos, endpos - pos, path ? path : "");
	}

	return result;
}

void ThePEGInterface::readParameterSet(const edm::ParameterSet &pset, const string &paramSet) const
{
	stringstream logstream;
	ofstream cfgDump;
	if (!configDump_.empty())
		cfgDump.open(configDump_.c_str(), ios_base::app);

	// Read CMSSW config file parameter set
	vector<string> params = pset.getParameter<vector<string> >(paramSet);

	// Loop over the parameter sets
	for(vector<string>::const_iterator psIter = params.begin();
	    psIter != params.end(); ++psIter) {

		// Include other parameter sets specified by +psName
		if (psIter->find_first_of('+') == 0) {
			edm::LogInfo("ThePEGInterface") << "Loading parameter set (" << psIter->substr(1) << ")";
			if (!configDump_.empty())
				cfgDump << endl << "####### " << psIter->substr(1) << " #######" << endl;
			readParameterSet(pset, psIter->substr(1));
		}
		// Topmost parameter set is called "parameterSets"
		else if (paramSet == "parameterSets") {
			edm::LogInfo("ThePEGInterface") << "Loading parameter set (" << *psIter << ")";
			if (!configDump_.empty())
				cfgDump << endl << "####### " << *psIter << " #######" << endl;
			readParameterSet(pset, *psIter);
		}
		// Transfer parameters to the repository
		else {
			string line = resolveEnvVars(*psIter);
			string out = ThePEG::Repository::exec(line, logstream);
			if (!configDump_.empty())
				cfgDump << line << endl;
			if (out != "")
				edm::LogInfo("ThePEGInterface") << line << " => " << out;
		}
	}
}

void ThePEGInterface::initRepository(const edm::ParameterSet &pset) const
{
	/* Initialize the repository from
	 * 1. the repository file (default values)
	 * 2. the ThePEG config files
	 * 3. the CMSSW config blocks
	 */
	stringstream logstream;

	// Read the repository of serialized default values
	string repository = dataFile(pset, "repository");
	if (!repository.empty()) {
		edm::LogInfo("ThePEGInterface") << "Loading repository (" << repository << ")";
		ThePEG::Repository::load(repository);
	}

	if (!getenv("ThePEG_INSTALL_PATH")) {
		vector<string> libdirlist = ThePEG::DynamicLoader::allPaths();
		for(vector<string>::const_iterator libdir = libdirlist.begin();
		    libdir < libdirlist.end(); ++libdir) {
			if (libdir->empty() || (*libdir)[0] != '/')
				continue;
			if (boost::filesystem::exists(*libdir +
					"/../../share/ThePEG/PDFsets.index")) {
				setenv("ThePEG_INSTALL_PATH",
				       libdir->c_str(), 0);
				break;
			}
		}
	}

	// Read ThePEG config files to read
	vector<string> configFiles = pset.getParameter<vector<string> >("configFiles");

	// Loop over the config files
	for(vector<string>::const_iterator iter = configFiles.begin();
	    iter != configFiles.end(); ++iter) {
		edm::LogInfo("ThePEGInterface") << "Reading config file (" << *iter << ")";
                ThePEG::Repository::read(dataFile(*iter), logstream);
                edm::LogInfo("ThePEGSource") << logstream.str();
	}

	// Read CMSSW config file parameter sets starting from "parameterSets"
	readParameterSet(pset, "parameterSets");

	// Print the directories where ThePEG looks for libs
	vector<string> libdirlist = ThePEG::DynamicLoader::allPaths();
	for(vector<string>::const_iterator libdir = libdirlist.begin();
	    libdir < libdirlist.end(); ++libdir)
		edm::LogInfo("ThePEGInterface") << "DynamicLoader path = " << *libdir << endl;

	// Output status information about the repository
	ThePEG::Repository::stats(logstream);
	edm::LogInfo("ThePEGInterface") << logstream.str();
}

void ThePEGInterface::initGenerator()
{
	// Get generator from the repository and initialize it
	ThePEG::BaseRepository::CheckObjectDirectory(generator_);
	ThePEG::EGPtr tmp = ThePEG::BaseRepository::GetObject<ThePEG::EGPtr>(generator_);
	if (tmp) {
		eg_ = ThePEG::Repository::makeRun(tmp, run_);
		eg_->initialize();
		edm::LogInfo("ThePEGInterface") << "EventGenerator initialized";
	} else
		throw cms::Exception("ThePEGInterface")
			<< "EventGenerator could not be initialized!" << endl;

	// Skip events
	for (int i = 0; i < skipEvents_; i++) {
		eg_->shoot();
		edm::LogInfo("ThePEGInterface") << "Event discarded";
	}
}

auto_ptr<HepMC::GenEvent> ThePEGInterface::convert(
					const ThePEG::EventPtr &event)
{
	return std::auto_ptr<HepMC::GenEvent>(
		ThePEG::HepMCConverter<HepMC::GenEvent>::convert(*event));
}

void ThePEGInterface::clearAuxiliary(HepMC::GenEvent *hepmc,
                                     HepMC::PdfInfo *pdf)
{
	if (hepmc) {
		hepmc->set_event_scale(-1.0);
		hepmc->set_alphaQCD(-1.0);
		hepmc->set_alphaQED(-1.0);
	}
	if (pdf) {
		pdf->set_id1(-100);
		pdf->set_id2(-100);
		pdf->set_x1(-1.0);
		pdf->set_x2(-1.0);
		pdf->set_scalePDF(-1.0);
		pdf->set_pdf1(-1.0);
		pdf->set_pdf2(-1.0);
	}
}

void ThePEGInterface::fillAuxiliary(HepMC::GenEvent *hepmc,
                                    HepMC::PdfInfo *pdf,
                                    const ThePEG::EventPtr &event)
{
	if (!event->primaryCollision())
		return;

	ThePEG::tcEHPtr eh = ThePEG::dynamic_ptr_cast<ThePEG::tcEHPtr>(
				event->primaryCollision()->handler());
	double scale = eh->lastScale();

	if (hepmc) {
		if (hepmc->event_scale() < 0 && scale > 0)
			hepmc->set_event_scale(std::sqrt(scale) / ThePEG::GeV);

		if (hepmc->alphaQCD() < 0)
			hepmc->set_alphaQCD(eh->lastAlphaS());
		if (hepmc->alphaQED() < 0)
			hepmc->set_alphaQED(eh->lastAlphaEM());
	}

	if (pdf) {
		const ThePEG::PPair &beams = eh->lastParticles();
		const ThePEG::PPair &partons = eh->lastPartons();
		ThePEG::tcPDFPtr pdf1 = eh->lastExtractor()->getPDF(
						beams.first->dataPtr());
		ThePEG::tcPDFPtr pdf2 = eh->lastExtractor()->getPDF(
						beams.second->dataPtr());
		double x1, x2;

		if (pdf->id1() == -100) {
			int id = partons.first->id();
			pdf->set_id1(id == 21 ? 0 : id);
		}
		if (pdf->id2() == -100) {
			int id = partons.second->id();
			pdf->set_id2(id == 21 ? 0 : id);
		}

		if (pdf->scalePDF() < 0)
			pdf->set_scalePDF(std::sqrt(scale) / ThePEG::GeV);
		else
			scale = ThePEG::sqr(pdf->scalePDF()) * ThePEG::GeV;

		if (pdf->x1() < 0) {
			x1 = eh->lastX1();
			pdf->set_x1(x1);
		} else
			x1 = pdf->x1();

		if (pdf->x2() < 0) {
			x2 = eh->lastX2();
			pdf->set_x2(x2);
		} else
			x2 = pdf->x2();

		if (pdf1 && pdf->pdf1() < 0) {
			double v = pdf1->xfx(beams.first->dataPtr(),
			                     partons.first->dataPtr(),
			                     scale, x1);
			if (v > 0)
				pdf->set_pdf1(v);
			else
				pdf->set_pdf2(-1.0);
		}
		if (pdf2 && pdf->pdf2() < 0) {
			double v = pdf2->xfx(beams.first->dataPtr(),
			                     partons.first->dataPtr(),
			                     scale, x2);
			if (v > 0)
				pdf->set_pdf2(v);
			else
				pdf->set_pdf2(-1.0);
		}
	}

}
