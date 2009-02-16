#include <memory>

#include <HepMC/GenEvent.h>
#include <HepMC/IO_BaseClass.h>

#include <ThePEG/Repository/Repository.h>
#include <ThePEG/EventRecord/Event.h>
#include <ThePEG/Config/ThePEG.h>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"

#include "GeneratorInterface/Core/interface/GeneratorFilter.h"
#include "GeneratorInterface/Core/interface/HadronizerFilter.h"

#include "GeneratorInterface/ThePEGInterface/interface/ThePEGInterface.h"

class ThePEGHadronizer : public ThePEGInterface {
    public:
	ThePEGHadronizer(const edm::ParameterSet &params);
	virtual ~ThePEGHadronizer();

	bool initializeForInternalPartons();
//	bool initializeForExternalPartons();
	bool declareStableParticles(const std::vector<int> &pdgIds);

	void statistics();

	bool generatePartonsAndHadronize();
//	bool hadronize();

	bool decay();
	void resetEvent(const HepMC::GenEvent *e)	// WHAT THE FUCK?!
	{ genEvent.reset(const_cast<HepMC::GenEvent*>(e)); }	// WTF^2
	bool residualDecay();
	void finalizeEvent();

	const GenRunInfoProduct &getGenRunInfo() const { return genRunInfo; }
	HepMC::GenEvent *getGenEvent() { return genEvent.release(); }

	const char *classname() const { return "ThePEGHadronizer"; }

    private:
	unsigned int			eventsToPrint;
	unsigned int			index;

	GenRunInfoProduct		genRunInfo;
	ThePEG::EventPtr		thepegEvent;
	std::auto_ptr<HepMC::GenEvent>	genEvent;
};

ThePEGHadronizer::ThePEGHadronizer(const edm::ParameterSet &pset) :
	ThePEGInterface(pset),
	eventsToPrint(pset.getUntrackedParameter<unsigned int>("eventsToPrint", 0)),
	index(0)
{  
	initRepository(pset);

	genRunInfo.setExternalXSecLO(
		pset.getUntrackedParameter<double>("crossSection", -1.0));
	genRunInfo.setFilterEfficiency(
		pset.getUntrackedParameter<double>("filterEfficiency", -1.0));
}

ThePEGHadronizer::~ThePEGHadronizer()
{
}

bool ThePEGHadronizer::initializeForInternalPartons()
{
	initGenerator();
	return true;
}

bool ThePEGHadronizer::declareStableParticles(const std::vector<int> &pdgIds)
{
	return false;
}

void ThePEGHadronizer::statistics()
{
	genRunInfo.setInternalXSec(GenRunInfoProduct::XSec(
				eg_->integratedXSec() / ThePEG::picobarn));
}

bool ThePEGHadronizer::generatePartonsAndHadronize()
{
	edm::LogInfo("Generator|ThePEGHadronizer") << "Start production";

	flushRandomNumberGenerator();

	thepegEvent = eg_->shoot();
	if (!thepegEvent) {
		edm::LogWarning("Generator|ThePEGHadronizer") << "thepegEvent not initialized";
		return false;
	}

	genEvent = convert(thepegEvent);
	if (!genEvent.get()) {
		edm::LogWarning("Generator|ThePEGHadronizer") << "genEvent not initialized";
		return false;
	}

	return true;
}

void ThePEGHadronizer::finalizeEvent()
{
	HepMC::PdfInfo pdf;
	clearAuxiliary(genEvent.get(), &pdf);
	genEvent->set_event_number(++index);
	fillAuxiliary(genEvent.get(), &pdf, thepegEvent);
	genEvent->set_pdf_info(pdf);

	if (eventsToPrint) {
		eventsToPrint--;
		genEvent->print();
	}

	if (iobc_.get())
		iobc_->write_event(genEvent.get());

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

typedef edm::GeneratorFilter<ThePEGHadronizer> ThePEGGeneratorFilter;
DEFINE_FWK_MODULE(ThePEGGeneratorFilter);

#if 0
typedef edm::HadronizerFilter<ThePEGHadronizer> ThePEGHadronizerFilter;
DEFINE_FWK_MODULE(ThePEGHadronizerFilter);
#endif
