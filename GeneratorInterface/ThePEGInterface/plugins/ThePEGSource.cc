#include <memory>

#include <HepMC/GenEvent.h>
#include <HepMC/IO_BaseClass.h>

#include <ThePEG/Repository/Repository.h>
#include <ThePEG/EventRecord/Event.h>
#include <ThePEG/Config/ThePEG.h>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/GeneratedInputSource.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"

#include "GeneratorInterface/ThePEGInterface/interface/ThePEGInterface.h"

class ThePEGSource : public edm::GeneratedInputSource, public ThePEGInterface {
    public:
	ThePEGSource(const edm::ParameterSet &params,
	             const edm::InputSourceDescription &desc);
	virtual ~ThePEGSource();

    private:
	virtual void beginRun(edm::Run &run);
	virtual void endRun(edm::Run &run);
	virtual bool produce(edm::Event &event);

	unsigned int	eventsToPrint;

	const double	extCrossSect;
	const double	extFilterEff;
};

ThePEGSource::ThePEGSource(const edm::ParameterSet &pset,
                           edm::InputSourceDescription const &desc) :
	edm::GeneratedInputSource(pset, desc),
	ThePEGInterface(pset),
	eventsToPrint(pset.getUntrackedParameter<unsigned int>("eventsToPrint", 0)),
	extCrossSect(pset.getUntrackedParameter<double>("crossSection", -1.0)),
	extFilterEff(pset.getUntrackedParameter<double>("filterEfficiency", -1.0))
{  
	initRepository(pset);

	produces<edm::HepMCProduct>();
	produces<GenRunInfoProduct, edm::InRun>();
}

ThePEGSource::~ThePEGSource()
{
}

void ThePEGSource::beginRun(edm::Run &run)
{
	initGenerator();
}

void ThePEGSource::endRun(edm::Run &run)
{
	std::auto_ptr<GenRunInfoProduct> genRunInfo(new GenRunInfoProduct);
	genRunInfo->setInternalXSec(eg_->integratedXSec() / ThePEG::picobarn);
	genRunInfo->setExternalXSecLO(extCrossSect);
	genRunInfo->setFilterEfficiency(extFilterEff);
	run.put(genRunInfo);
}

bool ThePEGSource::produce(edm::Event &event)
{
	edm::LogInfo("Generator|ThePEGSource") << "Start production";

	flushRandomNumberGenerator();
	ThePEG::EventPtr thepegEvent = eg_->shoot();
	if (!thepegEvent) {
		edm::LogWarning("Generator|ThePEGSource") << "thepegEvent not initialized";
		return false;
	}

	std::auto_ptr<HepMC::GenEvent> hepmcEvent = convert(thepegEvent);
	if (!hepmcEvent.get()) {
		edm::LogWarning("Generator|ThePEGSource") << "hepmcEvent not initialized";
		return false;
	}

	HepMC::PdfInfo pdf;
	clearAuxiliary(hepmcEvent.get(), &pdf);
	hepmcEvent->set_event_number(numberEventsInRun() -
	                             remainingEvents() - 1);
	fillAuxiliary(hepmcEvent.get(), &pdf, thepegEvent);
	if (usePthatEventScale)
		setPthatEventScale(hepmcEvent.get(), thepegEvent);
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
	edm::LogInfo("Generator|ThePEGSource") << "Event produced";

	return true;
}

DEFINE_ANOTHER_FWK_INPUT_SOURCE(ThePEGSource);
