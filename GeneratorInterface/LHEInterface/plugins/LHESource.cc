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

#include "GeneratorInterface/LHEInterface/interface/LesHouches.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
#include "GeneratorInterface/LHEInterface/interface/LHEReader.h"
#include "GeneratorInterface/LHEInterface/interface/LHECommonProduct.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEventProduct.h"

#include "LHESource.h"

using namespace lhef;

LHESource::LHESource(const edm::ParameterSet &params,
                     const edm::InputSourceDescription &desc) :
	GeneratedInputSource(params, desc),
	reader(new LHEReader(params)),
	skipEvents(params.getUntrackedParameter<unsigned int>("skipEvents", 0))
{
#if 0
	produces<LHEEventProduct>();
	produces<LHECommonProduct, edm::InRun>();
#else
	produces<HEPEUP>();
	produces<HEPRUP, edm::InRun>();
#endif
}

LHESource::LHESource(const edm::ParameterSet &params,
                     const edm::InputSourceDescription &desc,
                     lhef::LHEReader *reader) :
	GeneratedInputSource(params, desc),
	reader(reader),
	skipEvents(params.getUntrackedParameter<unsigned int>("skipEvents", 0))
{
#if 0
	produces<LHEEventProduct>();
	produces<LHECommonProduct, edm::InRun>();
#else
	produces<HEPEUP>();
	produces<HEPRUP, edm::InRun>();
#endif
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

	if (!heprup.get()) {
		heprup.reset(new HEPRUP(*partonLevel->getHEPRUP()));
		
	}

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
