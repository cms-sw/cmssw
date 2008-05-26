#include <algorithm>
#include <functional>
#include <iostream>
#include <string>
#include <memory>

#include <boost/bind.hpp>

#include "FWCore/Framework/interface/GeneratedInputSource.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"
#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
#include "GeneratorInterface/LHEInterface/interface/LHEReader.h"
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

#include "LHESource.h"

using namespace lhef;

LHESource::LHESource(const edm::ParameterSet &params,
                     const edm::InputSourceDescription &desc) :
	GeneratedInputSource(params, desc),
	reader(new LHEReader(params)),
	skipEvents(params.getUntrackedParameter<unsigned int>("skipEvents", 0))
{
	produces<LHEEventProduct>();
	produces<LHERunInfoProduct, edm::InRun>();
}

LHESource::LHESource(const edm::ParameterSet &params,
                     const edm::InputSourceDescription &desc,
                     lhef::LHEReader *reader) :
	GeneratedInputSource(params, desc),
	reader(reader),
	skipEvents(params.getUntrackedParameter<unsigned int>("skipEvents", 0))
{
	produces<LHEEventProduct>();
	produces<LHERunInfoProduct, edm::InRun>();
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
	if (partonLevel)
		return;

	while(skipEvents > 0) {
		skipEvents--;
		partonLevel = reader->next();
		if (!partonLevel)
			return;
	}

	partonLevel = reader->next();
	if (!partonLevel)
			return;

	if (!runInfo)
		runInfo = partonLevel->getRunInfo();
}

void LHESource::beginRun(edm::Run &run)
{
	nextEvent();
	if (runInfo) {
		std::auto_ptr<LHERunInfoProduct> product(
				new LHERunInfoProduct(*runInfo->getHEPRUP()));
		std::for_each(runInfo->getHeaders().begin(),
		              runInfo->getHeaders().end(),
		              boost::bind(
		              	&LHERunInfoProduct::addHeader,
		              	product.get(), _1));
		std::for_each(runInfo->getComments().begin(),
		              runInfo->getComments().end(),
		              boost::bind(&LHERunInfoProduct::addComment,
		              	product.get(), _1));
		run.put(product);
		runInfo.reset();

	}
}

bool LHESource::produce(edm::Event &event)
{
	nextEvent();
	if (!partonLevel)
		return false;

	std::auto_ptr<LHEEventProduct> product(
			new LHEEventProduct(*partonLevel->getHEPEUP()));
	if (partonLevel->getPDF())
		product->setPDF(*partonLevel->getPDF());
	std::for_each(partonLevel->getComments().begin(),
	              partonLevel->getComments().end(),
	              boost::bind(&LHEEventProduct::addComment,
	                          product.get(), _1));
	event.put(product);

	partonLevel.reset();
	return true;
}

DEFINE_ANOTHER_FWK_INPUT_SOURCE(LHESource);
