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
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenInfoProduct.h"

#include "GeneratorInterface/ThePEGInterface/interface/ThePEGInterface.h"

class ThePEGProducer : public edm::EDProducer, public ThePEGInterface {
    public:
	ThePEGProducer(const edm::ParameterSet &params);
	virtual ~ThePEGProducer();

    private:
	virtual void beginRun(edm::Run &run, const edm::EventSetup &es);
	virtual void endRun(edm::Run &run, const edm::EventSetup &es);
	virtual void produce(edm::Event &event, const edm::EventSetup &es);

	unsigned int	eventsToPrint;
	unsigned int	index;

	const double	extCrossSect;
	const double	extFilterEff;
};

ThePEGProducer::ThePEGProducer(const edm::ParameterSet &pset) :
	ThePEGInterface(pset),
	eventsToPrint(pset.getUntrackedParameter<unsigned int>("eventsToPrint", 0)),
	index(0),
	extCrossSect(pset.getUntrackedParameter<double>("crossSection", -1.0)),
	extFilterEff(pset.getUntrackedParameter<double>("filterEfficiency", -1.0))
{  
	initRepository(pset);

	produces<edm::HepMCProduct>();
	produces<edm::GenInfoProduct, edm::InRun>();
}

ThePEGProducer::~ThePEGProducer()
{
}

void ThePEGProducer::beginRun(edm::Run &run, const edm::EventSetup &es)
{
	initGenerator();

	edm::Service<edm::RandomNumberGenerator> rng;
	eg_->setSeed(rng->mySeed());
}

void ThePEGProducer::endRun(edm::Run &run, const edm::EventSetup &es)
{
	std::auto_ptr<edm::GenInfoProduct> genInfoProd(new edm::GenInfoProduct);
	genInfoProd->set_cross_section(eg_->integratedXSec() / ThePEG::picobarn);
	genInfoProd->set_external_cross_section(extCrossSect);
	genInfoProd->set_filter_efficiency(extFilterEff);
	run.put(genInfoProd);
}

void ThePEGProducer::produce(edm::Event &event, const edm::EventSetup &es)
{
	edm::LogInfo("Generator|ThePEGProducer") << "Start production";

	ThePEG::EventPtr thepegEvent = eg_->shoot();
	if (!thepegEvent) {
		edm::LogWarning("Generator|ThePEGProducer") << "thepegEvent not initialized";
		return;
	}

	std::auto_ptr<HepMC::GenEvent> hepmcEvent = convert(thepegEvent);
	if (!hepmcEvent.get()) {
		edm::LogWarning("Generator|ThePEGProducer") << "hepmcEvent not initialized";
		return;
	}

	HepMC::PdfInfo pdf;
	clearAuxiliary(hepmcEvent.get(), &pdf);
	hepmcEvent->set_event_number(++index);
	fillAuxiliary(hepmcEvent.get(), &pdf, thepegEvent);
	hepmcEvent->set_pdf_info(pdf);

	if (eventsToPrint) {
		eventsToPrint--;
		hepmcEvent->print();
	}

	if (iobc_.get())
		iobc_->write_event(hepmcEvent.get());

	std::auto_ptr<edm::HepMCProduct> result(new edm::HepMCProduct());
	result->addHepMCData(hepmcEvent.release());
	event.put(result);
	edm::LogInfo("Generator|ThePEGProducer") << "Event produced";
}

DEFINE_ANOTHER_FWK_MODULE(ThePEGProducer);
