/** \class ThePEGInterface
 *  $Id: ThePEGInterface.cc,v 1.16 2009/05/19 17:38:54 stober Exp $
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

#include <boost/shared_ptr.hpp>
#include <boost/filesystem.hpp>

#include <HepMC/GenEvent.h>
#include <HepMC/PdfInfo.h>
#include <HepMC/IO_GenEvent.h>

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

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "GeneratorInterface/Core/interface/ParameterCollector.h"

#include "GeneratorInterface/ThePEGInterface/interface/Proxy.h"
#include "GeneratorInterface/ThePEGInterface/interface/RandomEngineGlue.h"
#include "GeneratorInterface/ThePEGInterface/interface/HepMCConverter.h"
#include "GeneratorInterface/ThePEGInterface/interface/ThePEGInterface.h"

using namespace std;
using namespace gen;

ThePEGInterface::ThePEGInterface(const edm::ParameterSet &pset) :
	randomEngineGlueProxy_(ThePEG::RandomEngineGlue::Proxy::create()),
	dataLocation_(ParameterCollector::resolve(pset.getParameter<string>("dataLocation"))),
	generator_(pset.getParameter<string>("generatorModule")),
	run_(pset.getParameter<string>("run")),
	dumpConfig_(pset.getUntrackedParameter<string>("dumpConfig", "")),
	skipEvents_(pset.getUntrackedParameter<unsigned int>("skipEvents", 0))
{
	// Write events in hepmc ascii format for debugging purposes
	string dumpEvents = pset.getUntrackedParameter<string>("dumpEvents", "");
	if (!dumpEvents.empty()) {
		iobc_.reset(new HepMC::IO_GenEvent(dumpEvents.c_str(), ios::out));
		edm::LogInfo("ThePEGSource") << "Event logging switched on (=> " << dumpEvents << ")";
	}
	// Clear dumpConfig target
	if (!dumpConfig_.empty())
		ofstream cfgDump(dumpConfig_.c_str(), ios_base::trunc);
}

ThePEGInterface::~ThePEGInterface()
{
	if (eg_)
		eg_->finalize();
	edm::LogInfo("ThePEGInterface") << "Event generator finalized";
}

string ThePEGInterface::dataFile(const string &fileName) const
{
	if (fileName.empty() || fileName[0] == '/')
		return fileName;
	else
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

	ofstream cfgDump;
	ParameterCollector collector(pset);
	ParameterCollector::const_iterator iter;
	if (!dumpConfig_.empty()) {
		cfgDump.open(dumpConfig_.c_str(), ios_base::app);
		iter = collector.begin(cfgDump);
	} else
		iter = collector.begin();

	for(; iter != collector.end(); ++iter) {
		string out = ThePEG::Repository::exec(*iter, logstream);
		if (!out.empty()) {
			edm::LogInfo("ThePEGInterface") << *iter << " => " << out;
			cerr << "Error in ThePEG configuration!\n"
			        "\tLine: " << *iter << "\n" << out << endl;
		}
	}

	if (!dumpConfig_.empty()) {
		cfgDump << "saverun " << run_ << " " << generator_ << endl;
		cfgDump.close();
	}

	// write the ProxyID for the RandomEngineGlue to fill its pointer in
	ostringstream ss;
	ss << randomEngineGlueProxy_->getID();
	ThePEG::Repository::exec("set " + generator_ +
	                         ":RandomNumberGenerator:ProxyID " + ss.str(),
	                         logstream);

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
	for (unsigned int i = 0; i < skipEvents_; i++) {
		flushRandomNumberGenerator();
		eg_->shoot();
		edm::LogInfo("ThePEGInterface") << "Event discarded";
	}
}

void ThePEGInterface::flushRandomNumberGenerator()
{
	ThePEG::RandomEngineGlue *rnd = randomEngineGlueProxy_->getInstance();

	if (!rnd)
		edm::LogWarning("ProxyMissing")
			<< "ThePEG not initialised with RandomEngineGlue.";
	else
		rnd->flush();
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
			if (v > 0 && x1 > 0)
				pdf->set_pdf1(v / x1);
			else
				pdf->set_pdf2(-1.0);
		}
		if (pdf2 && pdf->pdf2() < 0) {
			double v = pdf2->xfx(beams.first->dataPtr(),
			                     partons.first->dataPtr(),
			                     scale, x2);
			if (v > 0 && x2 > 0)
				pdf->set_pdf2(v / x2);
			else
				pdf->set_pdf2(-1.0);
		}
	}

}

double ThePEGInterface::pthat(const ThePEG::EventPtr &event)
{
	using namespace ThePEG;

	if (!event->primaryCollision())
		return -1.0;

	tSubProPtr sub = event->primaryCollision()->primarySubProcess();
	TmpTransform<tSubProPtr> tmp(sub, Utilities::getBoostToCM(
							sub->incoming()));

	double pthat = (*sub->outgoing().begin())->momentum().perp();
	for(PVector::const_iterator it = sub->outgoing().begin();
	    it != sub->outgoing().end(); ++it)
		pthat = std::min(pthat, (*it)->momentum().perp());

	return pthat / ThePEG::GeV;
}
