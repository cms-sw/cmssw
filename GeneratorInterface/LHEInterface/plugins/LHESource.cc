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

#include "GeneratorInterface/LHEInterface/interface/LesHouches.h"
#include "GeneratorInterface/LHEInterface/interface/LHECommon.h"
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
	produces<LHEEventProduct>();
	produces<LHECommonProduct, edm::InRun>();
}

LHESource::LHESource(const edm::ParameterSet &params,
                     const edm::InputSourceDescription &desc,
                     lhef::LHEReader *reader) :
	GeneratedInputSource(params, desc),
	reader(reader),
	skipEvents(params.getUntrackedParameter<unsigned int>("skipEvents", 0))
{
	produces<LHEEventProduct>();
	produces<LHECommonProduct, edm::InRun>();
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

	if (!common)
		common = partonLevel->getCommon();
}

void LHESource::beginRun(edm::Run &run)
{
	nextEvent();
	if (common) {
		std::auto_ptr<LHECommonProduct> product(
				new LHECommonProduct(*common->getHEPRUP()));
		std::for_each(common->getHeaders().begin(),
		              common->getHeaders().end(),
		              boost::bind(
		              	&LHECommonProduct::addHeader,
		              	product.get(), _1));
		run.put(product);
		common.reset();
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
	event.put(product);

	partonLevel.reset();
	return true;
}

DEFINE_ANOTHER_FWK_INPUT_SOURCE(LHESource);
