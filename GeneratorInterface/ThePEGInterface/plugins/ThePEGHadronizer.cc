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
#include "SimDataFormats/GeneratorProducts/interface/GenInfoProduct.h"

#include "GeneratorInterface/Core/interface/GeneratorFilter.h"
#include "GeneratorInterface/Core/interface/HadronizerFilter.h"

#include "GeneratorInterface/ThePEGInterface/interface/ThePEGInterface.h"

class ThePEGHadronizer : public ThePEGInterface {
    public:
	ThePEGHadronizer(const edm::ParameterSet &params);
	virtual ~ThePEGHadronizer();

	bool initializeForInternalPartons();
//	bool initializeForExternalPartons();

	bool generatePartonsAndHadronize();
//	bool hadronize();

	bool decay();
	bool declareStableParticles();

	void statistics();

	const edm::GenInfoProduct &getGenInfoProduct() const { return genInfoProd; }
	HepMC::GenEvent *getGenEvent() { return genEvent.release(); }

	const char *classname() const { return "ThePEGHadronizer"; }

    private:
	unsigned int			eventsToPrint;
	unsigned int			index;

	edm::GenInfoProduct		genInfoProd;
	std::auto_ptr<HepMC::GenEvent>	genEvent;
};

ThePEGHadronizer::ThePEGHadronizer(const edm::ParameterSet &pset) :
	ThePEGInterface(pset),
	eventsToPrint(pset.getUntrackedParameter<unsigned int>("eventsToPrint", 0)),
	index(0)
{  
	initRepository(pset);

	genInfoProd.set_external_cross_section(
		pset.getUntrackedParameter<double>("crossSection", -1.0));
	genInfoProd.set_filter_efficiency(
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

void ThePEGHadronizer::statistics()
{
	genInfoProd.set_cross_section(eg_->integratedXSec() / ThePEG::picobarn);
}

bool ThePEGHadronizer::generatePartonsAndHadronize()
{
	edm::LogInfo("Generator|ThePEGHadronizer") << "Start production";

	ThePEG::EventPtr thepegEvent = eg_->shoot();
	if (!thepegEvent) {
		edm::LogWarning("Generator|ThePEGHadronizer") << "thepegEvent not initialized";
		return false;
	}

	genEvent = convert(thepegEvent);
	if (!genEvent.get()) {
		edm::LogWarning("Generator|ThePEGHadronizer") << "genEvent not initialized";
		return false;
	}

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
	return true;
}

bool ThePEGHadronizer::decay()
{
	return true;
}

bool ThePEGHadronizer::declareStableParticles()
{
	return true;
}

typedef edm::GeneratorFilter<ThePEGHadronizer> ThePEGGeneratorFilter;
DEFINE_FWK_MODULE(ThePEGGeneratorFilter);

#if 0
typedef edm::HadronizerFilter<ThePEGHadronizer> ThePEGHadronizerFilter;
DEFINE_FWK_MODULE(ThePEGHadronizerFilter);
#endif
