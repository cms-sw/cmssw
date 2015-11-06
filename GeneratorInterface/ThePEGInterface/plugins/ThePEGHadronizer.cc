#include <memory>
#include <sstream>

#include <HepMC/GenEvent.h>
#include <HepMC/IO_BaseClass.h>

#include <ThePEG/Repository/Repository.h>
#include <ThePEG/EventRecord/Event.h>
#include <ThePEG/Config/ThePEG.h>
#include <ThePEG/LesHouches/LesHouchesReader.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"

#include "GeneratorInterface/Core/interface/BaseHadronizer.h"
#include "GeneratorInterface/Core/interface/GeneratorFilter.h"
#include "GeneratorInterface/Core/interface/HadronizerFilter.h"

#include "GeneratorInterface/LHEInterface/interface/LHEProxy.h"

#include "GeneratorInterface/ThePEGInterface/interface/ThePEGInterface.h"

namespace CLHEP {
  class HepRandomEngine;
}

class ThePEGHadronizer : public ThePEGInterface, public gen::BaseHadronizer {
    public:
	ThePEGHadronizer(const edm::ParameterSet &params);
	virtual ~ThePEGHadronizer();

	bool readSettings( int ) { return true; }
	bool initializeForInternalPartons();
	bool initializeForExternalPartons();
	bool declareStableParticles(const std::vector<int> &pdgIds);
	bool declareSpecialSettings( const std::vector<std::string> ) { return true; }

	void statistics();

	bool generatePartonsAndHadronize();
	bool hadronize();
	bool decay();
	bool residualDecay();
	void finalizeEvent();

	const char *classname() const { return "ThePEGHadronizer"; }

    private:

        virtual void doSetRandomEngine(CLHEP::HepRandomEngine* v) override { setPEGRandomEngine(v); }

	unsigned int			eventsToPrint;

	ThePEG::EventPtr		thepegEvent;
	
	boost::shared_ptr<lhef::LHEProxy> proxy_;
	const std::string		handlerDirectory_;
};

ThePEGHadronizer::ThePEGHadronizer(const edm::ParameterSet &pset) :
	ThePEGInterface(pset),
	BaseHadronizer(pset),
	eventsToPrint(pset.getUntrackedParameter<unsigned int>("eventsToPrint", 0)),
	handlerDirectory_(pset.getParameter<std::string>("eventHandlers"))
{  
	initRepository(pset);

}

ThePEGHadronizer::~ThePEGHadronizer()
{
}

bool ThePEGHadronizer::initializeForInternalPartons()
{
	initGenerator();
	return true;
}

bool ThePEGHadronizer::initializeForExternalPartons()
{
	proxy_ = lhef::LHEProxy::create();  
  
	std::ostringstream ss;
	ss << proxy_->getID();

	std::ostringstream logstream;
	ThePEG::Repository::exec("set " + handlerDirectory_ +
	                         "/LHEReader:ProxyID " + ss.str(), logstream);
	edm::LogInfo("Generator|LHEInterface") << logstream.str();	
	
	proxy_->loadRunInfo(getLHERunInfo());	
	
	initGenerator();
	return true;
}

bool ThePEGHadronizer::declareStableParticles(const std::vector<int> &pdgIds)
{
	return false;
}

void ThePEGHadronizer::statistics()
{
	runInfo().setInternalXSec(GenRunInfoProduct::XSec(
		eg_->integratedXSec() / ThePEG::picobarn,
		eg_->integratedXSecErr() / ThePEG::picobarn));
}

bool ThePEGHadronizer::generatePartonsAndHadronize()
{
	edm::LogInfo("Generator|ThePEGHadronizer") << "Start production";

	flushRandomNumberGenerator();

        try {
                thepegEvent = eg_->shoot();
        } catch (std::exception& exc) {
                edm::LogWarning("Generator|ThePEGHadronizer") << "EGPtr::shoot() thrown an exception, event skipped: " << exc.what();
                return false;
        } catch (...) {
                edm::LogWarning("Generator|ThePEGHadronizer") << "EGPtr::shoot() thrown an unknown exception, event skipped";
                return false;
        }        
        
	if (!thepegEvent) {
		edm::LogWarning("Generator|ThePEGHadronizer") << "thepegEvent not initialized";
		return false;
	}

	event() = convert(thepegEvent);
	if (!event().get()) {
		edm::LogWarning("Generator|ThePEGHadronizer") << "genEvent not initialized";
		return false;
	}

	return true;
}

bool ThePEGHadronizer::hadronize()
{
	edm::LogInfo("Generator|ThePEGHadronizer") << "Start production";

	flushRandomNumberGenerator();
	
	//need to copy lhe event here unfortunately because of interface mismatch
	proxy_->loadEvent(boost::shared_ptr<lhef::LHEEvent>(new lhef::LHEEvent(*lheEvent())));

        //dummy for now
        double mergeweight = 1.0;
        
	try {
		thepegEvent = eg_->shoot();
	} catch (std::exception& exc) {
		edm::LogWarning("Generator|ThePEGHadronizer") << "EGPtr::shoot() thrown an exception, event skipped: " << exc.what();
                lheEvent()->count( lhef::LHERunInfo::kSelected, 1.0, mergeweight );
		return false;
	} catch (...) {
		edm::LogWarning("Generator|ThePEGHadronizer") << "EGPtr::shoot() thrown an unknown exception, event skipped";
                lheEvent()->count( lhef::LHERunInfo::kSelected, 1.0, mergeweight );
		return false;
	}

	if (!thepegEvent) {
		edm::LogWarning("Generator|ThePEGHadronizer") << "thepegEvent not initialized";
                lheEvent()->count( lhef::LHERunInfo::kSelected, 1.0, mergeweight );
		return false;
	}

	event() = convert(thepegEvent);
	if (!event().get()) {
		edm::LogWarning("Generator|ThePEGHadronizer") << "genEvent not initialized";
                lheEvent()->count( lhef::LHERunInfo::kSelected, 1.0, mergeweight );
		return false;
	}
	
	//Fill LHE weight (since it's not otherwise propagated)
	event()->weights()[0] *= lheEvent()->getHEPEUP()->XWGTUP;

	HepMC::PdfInfo pdf;
	clearAuxiliary(event().get(), &pdf);
	lheEvent()->fillPdfInfo(&pdf);
	fillAuxiliary(event().get(), &pdf, thepegEvent);
	event()->set_pdf_info(pdf);

        // update LHE matching statistics
        //
        lheEvent()->count( lhef::LHERunInfo::kAccepted, 1.0, mergeweight );        
        
	return true;

}

void ThePEGHadronizer::finalizeEvent()
{
	HepMC::PdfInfo pdf;
	clearAuxiliary(event().get(), &pdf);
	fillAuxiliary(event().get(), &pdf, thepegEvent);
	event()->set_pdf_info(pdf);

	eventInfo().reset(new GenEventInfoProduct(event().get()));
	eventInfo()->setBinningValues(
			std::vector<double>(1, pthat(thepegEvent)));

	if (eventsToPrint) {
		eventsToPrint--;
		event()->print();
	}

	if (iobc_.get())
		iobc_->write_event(event().get());

	edm::LogInfo("Generator|ThePEGHadronizer") << "Event produced";
}

bool ThePEGHadronizer::decay()
{
	return true;
}

bool ThePEGHadronizer::residualDecay()
{
	return true;
}

#include "GeneratorInterface/ExternalDecays/interface/ExternalDecayDriver.h"

typedef edm::GeneratorFilter<ThePEGHadronizer, gen::ExternalDecayDriver> ThePEGGeneratorFilter;
DEFINE_FWK_MODULE(ThePEGGeneratorFilter);

typedef edm::HadronizerFilter<ThePEGHadronizer, gen::ExternalDecayDriver> ThePEGHadronizerFilter;
DEFINE_FWK_MODULE(ThePEGHadronizerFilter);
