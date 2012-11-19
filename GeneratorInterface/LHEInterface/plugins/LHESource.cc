#include <algorithm>
#include <functional>
#include <iostream>
#include <string>
#include <memory>

#include <boost/bind.hpp>
#include <boost/ptr_container/ptr_deque.hpp>

#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/OrphanHandle.h"

#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
#include "GeneratorInterface/LHEInterface/interface/LHEReader.h"

#include "LHESource.h"

using namespace lhef;

LHESource::LHESource(const edm::ParameterSet &params,
                     const edm::InputSourceDescription &desc) :
	ProducerSourceFromFiles(params, desc, false),
	reader(new LHEReader(fileNames(), params.getUntrackedParameter<unsigned int>("skipEvents", 0))),
	wasMerged(false)
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

	partonLevel = reader->next();
	if (!partonLevel)
			return;

	boost::shared_ptr<LHERunInfo> runInfoThis = partonLevel->getRunInfo();
	if (runInfoThis != runInfoLast) {
		runInfo = runInfoThis;
		runInfoLast = runInfoThis;
	}
}

void LHESource::beginRun(edm::Run &run)
{
	nextEvent();
	if (runInfoLast) {
		runInfo = runInfoLast;

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

		// keep a copy around in case of merging
		runInfoProducts.push_back(new LHERunInfoProduct(*product));
		wasMerged = false;

		run.put(product);

		runInfo.reset();
	}
}

void LHESource::endRun(edm::Run &run)
{
	if (!runInfoProducts.empty()) {
		std::auto_ptr<LHERunInfoProduct> product(
					runInfoProducts.pop_front().release());
		run.put(product);
	}
}

bool LHESource::setRunAndEventInfo(edm::EventID&, edm::TimeValue_t&)
{
	nextEvent();
	if (!partonLevel)
		return false;
        return true;
}

void LHESource::produce(edm::Event &event)
{
	std::auto_ptr<LHEEventProduct> product(
			new LHEEventProduct(*partonLevel->getHEPEUP()));
	if (partonLevel->getPDF())
		product->setPDF(*partonLevel->getPDF());
	std::for_each(partonLevel->getComments().begin(),
	              partonLevel->getComments().end(),
	              boost::bind(&LHEEventProduct::addComment,
	                          product.get(), _1));
	event.put(product);

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

		if (!runInfoProducts.empty()) {
			runInfoProducts.front().mergeProduct(*product);
			if (!wasMerged) {
				runInfoProducts.pop_front();
				runInfoProducts.push_front(product);
				wasMerged = true;
			}
		}

		runInfo.reset();
	}

	partonLevel.reset();
}

DEFINE_FWK_INPUT_SOURCE(LHESource);
