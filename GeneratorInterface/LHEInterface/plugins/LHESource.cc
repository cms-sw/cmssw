#include <iostream>
#include <string>
#include <memory>

#include "FWCore/Framework/interface/GeneratedInputSource.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
#include "GeneratorInterface/LHEInterface/interface/LHEReader.h"

using namespace lhef;

class LHESource : public edm::GeneratedInputSource {
    public:
	explicit LHESource(const edm::ParameterSet &params,
	                   const edm::InputSourceDescription &desc);
	virtual ~LHESource();

    protected:
	virtual void endJob();
	virtual void beginRun(edm::Run &run);
	virtual bool produce(edm::Event &event);

    private:
	void nextEvent();

	std::auto_ptr<LHEReader>	reader;
	std::auto_ptr<HEPRUP>		heprup;
	std::auto_ptr<HEPEUP>		hepeup;

	unsigned int			skipEvents;
};

LHESource::LHESource(const edm::ParameterSet &params,
                     const edm::InputSourceDescription &desc) :
	GeneratedInputSource(params, desc),
	reader(new LHEReader(params)),
	skipEvents(params.getUntrackedParameter<unsigned int>("skipEvents", 0))
{
	produces<HEPEUP>();
	produces<HEPRUP, edm::InRun>();
}

LHESource::~LHESource()
{
}

void LHESource::endJob()
{
	reader.reset();
}

void LHESource::nextEvent()
{
	if (hepeup.get())
		return;

	boost::shared_ptr<LHEEvent> partonLevel;
	while(skipEvents > 0) {
		skipEvents--;
		partonLevel = reader->next();
		if (!partonLevel.get())
			return;
	}

	partonLevel = reader->next();
	if (!partonLevel.get())
			return;

	if (!heprup.get())
		heprup.reset(new HEPRUP(*partonLevel->getHEPRUP()));

	hepeup.reset(new HEPEUP(*partonLevel->getHEPEUP()));
}

void LHESource::beginRun(edm::Run &run)
{
	nextEvent();
	if (heprup.get())
		run.put(heprup);
}

bool LHESource::produce(edm::Event &event)
{
	nextEvent();
	if (!hepeup.get())
		return false;

	event.put(hepeup);
	return true;
}

DEFINE_ANOTHER_FWK_INPUT_SOURCE(LHESource);
