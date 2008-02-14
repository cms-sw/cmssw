#include <iostream>
#include <string>
#include <memory>

#include <boost/shared_ptr.hpp>

#include <HepMC/GenEvent.h>

#include "FWCore/Framework/interface/GeneratedInputSource.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
#include "GeneratorInterface/LHEInterface/interface/LHEReader.h"
#include "GeneratorInterface/LHEInterface/interface/Hadronisation.h"

using namespace lhef;

class LHESource : public edm::GeneratedInputSource {
    public:
	LHESource(const edm::ParameterSet &params,
	          const edm::InputSourceDescription &desc);
	virtual ~LHESource();

    private:
	virtual bool produce(edm::Event &event);

	LHEReader			reader;
	std::auto_ptr<Hadronisation>	hadronisation;
};

LHESource::LHESource(const edm::ParameterSet &params,
                     const edm::InputSourceDescription &desc) :
	GeneratedInputSource(params, desc),
	reader(params),
	hadronisation(Hadronisation::create(
		params.getParameter<edm::ParameterSet>("hadronisation")))
{
	produces<edm::HepMCProduct>();
}

LHESource::~LHESource()
{
}

bool LHESource::produce(edm::Event &event)
{
	boost::shared_ptr<LHEEvent> partonLevel = reader.next();
	if (!partonLevel.get())
		return false;

	hadronisation->setEvent(partonLevel);

	std::auto_ptr<HepMC::GenEvent> hadronLevel =
					hadronisation->hadronize();

	hadronLevel->set_event_number(numberEventsInRun() -
	                              remainingEvents() - 1);

	std::auto_ptr<edm::HepMCProduct> result(new edm::HepMCProduct());
	result->addHepMCData(hadronLevel.release());
	event.put(result);

	return true;
}

DEFINE_ANOTHER_FWK_INPUT_SOURCE(LHESource);
