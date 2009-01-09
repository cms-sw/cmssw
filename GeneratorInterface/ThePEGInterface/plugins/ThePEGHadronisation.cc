#include <string>
#include <iostream>
#include <sstream>
#include <memory>
#include <assert.h>

#include <boost/shared_ptr.hpp>

#include <HepMC/GenEvent.h>
#include <HepMC/GenParticle.h>
#include <HepMC/PdfInfo.h>
#include <HepMC/IO_ExtendedAscii.h>
#include <HepMC/IO_BaseClass.h>

#include <ThePEG/Repository/Repository.h>
#include <ThePEG/EventRecord/Event.h>
#include <ThePEG/Config/ThePEG.h>
#include <ThePEG/LesHouches/LesHouchesReader.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Framework/interface/Event.h"

#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
#include "GeneratorInterface/LHEInterface/interface/LHEProxy.h"
#include "GeneratorInterface/LHEInterface/interface/Hadronisation.h"

#include "GeneratorInterface/ThePEGInterface/interface/ThePEGInterface.h"

using namespace std;
using namespace ThePEG;
using namespace lhef;

namespace lhef {

class ThePEGHadronisation : public ThePEGInterface, public Hadronisation {
    public:
	ThePEGHadronisation(const edm::ParameterSet &params);
	~ThePEGHadronisation();

    private:
	void doInit();
	std::auto_ptr<HepMC::GenEvent> doHadronisation();
	void newRunInfo(const boost::shared_ptr<LHERunInfo> &runInfo);

	void initLHE();

	const string			handlerDirectory_;

	boost::shared_ptr<LHEProxy>	proxy_;
};

void ThePEGHadronisation::initLHE()
{
	ostringstream ss;
	ss << proxy_->getID();

	ostringstream logstream;

	ThePEG::Repository::exec("cd " + handlerDirectory_, logstream);
	ThePEG::Repository::exec("set LHEReader:ProxyID " + ss.str(), logstream);

	edm::LogInfo("Generator|LHEInterface") << logstream.str();
}
  
ThePEGHadronisation::ThePEGHadronisation(const edm::ParameterSet &params) :
	ThePEGInterface(params),
	Hadronisation(params),
	handlerDirectory_(params.getParameter<string>("eventHandlers"))
{
	initRepository(params);

	proxy_ = LHEProxy::create();

	initLHE();
}

void ThePEGHadronisation::doInit()
{
}

ThePEGHadronisation::~ThePEGHadronisation()
{
}

std::auto_ptr<HepMC::GenEvent> ThePEGHadronisation::doHadronisation()
{
	edm::LogInfo("Generator|LHEInterface") << "Start production";

	proxy_->loadEvent(getRawEvent());

	ThePEG::EventPtr thepegEvent;
	try {
		thepegEvent = eg_->shoot();
	} catch(ThePEG::Stop) {
		// no event
	}

	if (!thepegEvent) {
		edm::LogWarning("Generator|LHEInterface")
			<< "thepegEvent not initialized";
		return std::auto_ptr<HepMC::GenEvent>();
	}

	std::auto_ptr<HepMC::GenEvent> event = convert(thepegEvent);
	if (!event.get())
		return event;

	HepMC::PdfInfo pdf;
	clearAuxiliary(event.get(), &pdf);
	getRawEvent()->fillPdfInfo(&pdf);
	fillAuxiliary(event.get(), &pdf, thepegEvent);
	event->set_pdf_info(pdf);

	return event;
}

void ThePEGHadronisation::newRunInfo(
				const boost::shared_ptr<LHERunInfo> &runInfo)
{
	proxy_->loadRunInfo(runInfo);

	initGenerator();

	edm::Service<edm::RandomNumberGenerator> rng;
	eg_->setSeed(rng->mySeed());
}

DEFINE_LHE_HADRONISATION_PLUGIN(ThePEGHadronisation);

} // namespace lhef
